import os
import json
import base64
import random
import datetime
from typing import Tuple, Optional
from dataclasses import dataclass
from fastapi import FastAPI, BackgroundTasks, Request, Body, HTTPException
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from uuid import uuid4 as uuid
import a2a
import redis
import chess
import chess.engine
from minio import Minio
from minio.error import S3Error

load_dotenv()
app = FastAPI()

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
CHESS_ENGINE_PATH = os.getenv("CHESS_ENGINE_PATH")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")
MINIO_BUCKET_ACCESS_KEY = os.getenv("MINIO_BUCKET_ACCESS_KEY")
MINIO_BUKCET_SECRET_KEY = os.getenv("MINIO_BUKCET_SECRET_KEY")

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_BUCKET_ACCESS_KEY,
    secret_key=MINIO_BUKCET_SECRET_KEY,
)


@dataclass
class RedisKeys:
    games = "games"


class RandomNameRepository:
    WORDS = [
        "sun", "moon", "tree", "cloud", "river", "stone", "eagle", "wolf", "fire",
        "wind", "storm", "leaf", "sky", "night", "light", "shadow", "mountain",
        "ocean", "echo", "whisper", "flame", "dust", "branch",
    ]

    @classmethod
    def generate_filename(cls, extension: str = "svg", word_count: int = 3) -> str:
        words = random.sample(cls.WORDS, word_count)
        return "_".join(words) + f".{extension}"

    @classmethod
    def generate_suffix(cls, word_count: int = 2) -> str:
        return "_".join(random.sample(cls.WORDS, word_count))

class Game:
    def __init__(self, board, engine, engine_time_limit=0.5):
        self.board = board
        self.engine = engine
        self.engine_time_limit = engine_time_limit

    def aimove(self):
        ai = self.engine.play(
            self.board, chess.engine.Limit(time=self.engine_time_limit)
        )
        self.board.push(ai.move)
        return ai.move, self.board

    def usermove(self, move):
        try:
            self.board.push_san(move)
        except ValueError:
            raise ValueError(f"Invalid move: {move}")
        return self.board

    def to_dict(self):
        return {"fen": self.board.fen(), "engine_time_limit": self.engine_time_limit}

    @classmethod
    def from_dict(cls, data):
        board = chess.Board(data["fen"])
        engine_time_limit = data.get("engine_time_limit", 0.5)
        engine = chess.engine.SimpleEngine.popen_uci(CHESS_ENGINE_PATH)
        return cls(board, engine, engine_time_limit)


class GameRepository:
    def __init__(self, redis_client, redis_key_prefix=RedisKeys.games):
        self.r = redis_client
        self.prefix = redis_key_prefix

    def _game_key(self, task_id: str) -> str:
        return f"{self.prefix}:{task_id}"

    def save(self, task_id: str, game: Game):
        key = self._game_key(task_id)
        self.r.set(key, json.dumps(game.to_dict()))

    def load(self, task_id: str) -> Optional[Game]:
        key = self._game_key(task_id)
        data = self.r.get(key)
        if data:
            return Game.from_dict(json.loads(data))
        return None

    def delete(self, task_id: str):
        key = self._game_key(task_id)
        self.r.delete(key)

    def start_game(self, engine_path: str) -> Game:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        board = chess.Board()

        return Game(board, engine)

    def parse_command(self, message: str) -> str:
        message = message.strip()
        message_lowercase = message.lower()

        if "resign" in message_lowercase:
            return "resign"
        if "board" in message_lowercase:
            return "board"

        return message


@app.get("/", response_class=HTMLResponse)
def read_root():
    return '<p style="font-size:40px">Chess bot A2A</p>'


def generate_board_image(board):
    filename = RandomNameRepository.generate_filename()
    destination_file = f"public/chessagent/{filename}"
    source_file = f"/tmp/{filename}"

    svg = board._repr_svg_()

    with open(source_file, "w") as f:
        f.write(svg)
        new_source_file = source_file.split(".svg")[0] + ".png"

        import cairosvg

        cairosvg.svg2png(url=source_file, write_to=new_source_file)

        source_file = new_source_file
        destination_file = destination_file.split(".svg")[0] + ".png"

    minio_client.fput_object(
        MINIO_BUCKET_NAME,
        destination_file,
        source_file,
    )

    image_url = f"https://media.tifi.tv/{MINIO_BUCKET_NAME}/{destination_file}"

    return image_url, filename


game_repo = GameRepository(r)


async def handle_task_send(params: a2a.TaskSendParams):
    session_id = params.sessionId
    task_id = params.id
    
    game = game_repo.load(task_id)

    if not game:
        game = game_repo.start_game(engine_path=CHESS_ENGINE_PATH)

    user_input = params.message.parts[0].text.strip()
    user_move = game_repo.parse_command(user_input)

    if user_move == "board":
        image_url, filename = generate_board_image(game.board)
        return a2a.SendTaskResponse(
            result=a2a.Task(
                id=params.id,
                sessionId=params.sessionId,
                status=a2a.TaskStatus(
                    state=a2a.TaskState.INPUT_REQUIRED,
                    message=a2a.Message(
                        role="agent",
                        parts=[
                            a2a.TextPart(text="Board state is:"),
                            a2a.FilePart(
                                file=a2a.FileContent(
                                    name=filename,
                                    mimeType="image/svg+xml",
                                    uri=image_url,
                                )
                            ),
                        ],
                    ),
                ),
            ),
        )        

    if user_move == "resign":
        return a2a.SendTaskResponse(
            result=a2a.Task(
                id=params.id,
                sessionId=params.sessionId,
                status=a2a.TaskStatus(
                    state=a2a.TaskState.COMPLETED,
                    message=a2a.Message(
                        role="agent",
                        parts=[
                            a2a.TextPart(text="Game ended by resignation."),
                            a2a.TextPart(
                                text="Start a new game by entering a valid move."
                            ),
                        ],
                    ),
                ),
            ),
        )

    try:
        game.usermove(user_move)
    except ValueError:
        response = a2a.SendTaskResponse(
            error=a2a.InvalidParamsError(
                message=f"Invalid move: '{user_move}'",
                data=f"You sent '{user_move}' which is not a valid chess move",
            ),
        )
        return response
    except:
        response = a2a.SendTaskResponse(
            error=a2a.InvalidParamsError(
                message="An error occured",
            ),
        )
        return response

    aimove, board = game.aimove()
    game_repo.save(task_id, game)
    image_url, filename = generate_board_image(board)

    if board.is_game_over():
        return a2a.SendTaskResponse(
            result=a2a.Task(
                id=task_id,
                sessionId=session_id,
                status=a2a.TaskStatus(
                    state=a2a.TaskState.COMPLETED,
                    message=a2a.Message(
                        role="agent",
                        parts=[
                            a2a.TextPart(text=f"Game over. AI moved {aimove.uci()}"),
                            a2a.FilePart(
                                file=a2a.FileContent(
                                    name=filename,
                                    mimeType="image/svg+xml",
                                    uri=image_url,
                                )
                            ),
                            a2a.TextPart(text="Start a new game by entering a valid move"),
                        ],
                    ),
                ),
            ),
        )

    response = a2a.SendTaskResponse(
        result=a2a.Task(
            id=task_id,
            sessionId=session_id,
            status=a2a.TaskStatus(
                state=a2a.TaskState.WORKING,
                message=a2a.Message(
                    role="agent",
                    parts=[
                        a2a.TextPart(text=f"AI moved {aimove.uci()}"),
                        # a2a.TextPart(text=str(board)),
                        a2a.FilePart(
                            file=a2a.FileContent(
                                name=filename,
                                mimeType="image/svg+xml",
                                uri=image_url,
                            )
                        ),
                    ],
                ),
            ),
        ),
    )

    return response


async def handle_get_task(params: a2a.TaskQueryParams):
    response = a2a.GetTaskResponse(
        result=a2a.Task(
            id="task_id",
            sessionId="session_id",
            status=a2a.TaskStatus(
                state=a2a.TaskState.COMPLETED,
                message=a2a.Message(
                    role="agent",
                    parts=[
                        a2a.TextPart(text=f"Completed the stuff"),
                    ],
                ),
            ),
        ),
    )

    return response

@app.post("/")
async def handle_rpc(request_data: dict):
    try:
        # Parse the request using the TypeAdapter
        rpc_request = a2a.A2ARequest.validate_python(request_data)

        if isinstance(rpc_request, a2a.SendTaskRequest):
            print("tasks/send")
            return await handle_task_send(params=rpc_request.params)
        elif isinstance(rpc_request, a2a.GetTaskRequest):
            print("tasks/get")
            return await handle_get_task(params=rpc_request.params)
        else:
            raise HTTPException(status_code=400, detail="Method not supported")
            
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=400, detail="Could not handle task")

agent_name_suffix = (
    "-" + os.getenv("APP_ENV") + "_" + RandomNameRepository.generate_suffix()
    if os.getenv("APP_ENV") == "local"
    else ""
)

@app.get("/.well-known/agent.json")
def agent_card(request: Request):
    external_base = request.headers.get("x-external-base-url", "")
    base_url = str(request.base_url).rstrip("/") + external_base
    card = a2a.AgentCard(
        name=f"Chess Agent {agent_name_suffix}",
        description="An agent that plays chess. Accepts moves in standard notation and returns updated board state as FEN and an image.",
        url=f"{base_url}",
        provider=a2a.AgentProvider(
            organization="CoolVicradon",
            url=f"{base_url}/provider",
        ),
        version="1.0.0",
        documentationUrl=f"{base_url}/docs",
        capabilities=a2a.AgentCapabilities(
            streaming=False,
            pushNotifications=False,
            stateTransitionHistory=True,
        ),
        authentication=a2a.AgentAuthentication(schemes=["Bearer"]),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["application/x-fen", "image/png"],
        skills=[
            a2a.AgentSkill(
                id="play_move",
                name="Play Move",
                description="Plays a move and returns the updated board in FEN format and as an image.",
                tags=["chess", "gameplay", "board"],
                examples=["e4", "Nf3", "d5"],
                inputModes=["text/plain"],
                outputModes=["application/x-fen", "image/png"],
            )
        ],
    )

    return card


if __name__ == "__main__":
    import uvicorn

    PORT = int(os.getenv("PORT", 7000))

    uvicorn.run("main:app", host="127.0.0.1", port=PORT, reload=True)