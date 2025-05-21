from pydantic import BaseModel, UUID4, Field, model_validator
from typing import List, Optional, Dict, Union, Literal, NamedTuple, Any
from enum import Enum
from uuid import uuid4


class ErrorDetail(NamedTuple):
    message: str
    description: str

ERROR_CODES = {
    -32700: ErrorDetail("JSON parse error", "Invalid JSON was sent"),
    -32600: ErrorDetail("Invalid Request", "Request payload validation error"),
    -32601: ErrorDetail("Method not found", "Not a valid method"),
    -32602: ErrorDetail("Invalid params", "Invalid method parameters"),
    -32603: ErrorDetail("Internal error", "Internal JSON-RPC error"),
    -32001: ErrorDetail("Task not found", "Task not found with the provided id"),
    -32002: ErrorDetail("Task cannot be canceled", "Task cannot be canceled by the remote agent"),
    -32003: ErrorDetail("Push notifications not supported", "Push Notification is not supported by the agent"),
    -32004: ErrorDetail("Unsupported operation", "Operation is not supported"),
    -32005: ErrorDetail("Incompatible content types", "Incompatible content types between client and an agent"),
}

class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Any | None = None

class JSONParseError(JSONRPCError):
    code: int = -32700
    message: str = 'Invalid JSON payload'
    data: Any | None = None


class InvalidRequestError(JSONRPCError):
    code: int = -32600
    message: str = 'Request payload validation error'
    data: Any | None = None


class MethodNotFoundError(JSONRPCError):
    code: int = -32601
    message: str = 'Method not found'
    data: None = None


class InvalidParamsError(JSONRPCError):
    code: int = -32602
    message: str = 'Invalid parameters'
    data: Any | None = None


class InternalError(JSONRPCError):
    code: int = -32603
    message: str = 'Internal error'
    data: Any | None = None


class TaskNotFoundError(JSONRPCError):
    code: int = -32001
    message: str = 'Task not found'
    data: None = None


class TaskNotCancelableError(JSONRPCError):
    code: int = -32002
    message: str = 'Task cannot be canceled'
    data: None = None


class PushNotificationNotSupportedError(JSONRPCError):
    code: int = -32003
    message: str = 'Push Notification is not supported'
    data: None = None


class UnsupportedOperationError(JSONRPCError):
    code: int = -32004
    message: str = 'This operation is not supported'
    data: None = None


class ContentTypeNotSupportedError(JSONRPCError):
    code: int = -32005
    message: str = 'Incompatible content types'
    data: None = None




class TextPart(BaseModel):
    type: str = "text"
    text: str
    metadata: Optional[Dict[str, object]] = None

class FileContent(BaseModel):
    name: Optional[str] = None
    mimeType: Optional[str] = None
    bytes: Optional[str] = Field(default=None, description="Base64 encoded content")
    uri: Optional[str] = Field(default=None, description="URI to the file")

    @model_validator(mode='after')
    def check_bytes_or_uri(self) -> 'FileContent':
        if self.bytes and self.uri:
            raise ValueError("Only one of 'bytes' or 'uri' must be provided, not both.")
        if not self.bytes and not self.uri:
            raise ValueError("One of 'bytes' or 'uri' must be provided.")
        return self

class FilePart(BaseModel):
    type: str = "file"
    file: FileContent
    metadata: Optional[Dict[str, object]] = None

class DataPart(BaseModel):
    type: str = "data"
    data: Dict[str, object]
    metadata: Optional[Dict[str, object]] = None


Part = Union[TextPart, FilePart, DataPart]

class TaskState(Enum):
    submitted = "submitted"
    working = "working"
    inputrequired = "input-required"
    completed = "completed"
    canceled = "canceled"
    failed = "failed"
    unknown = "unknown"

class Message(BaseModel):
    role: Literal["agent", "user"]
    parts: List[Part]
    metadata: Optional[Dict[str, object]] = {}


class Artifact(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parts: List[Part]
    metadata: Optional[Dict[str, object]] = {}
    index: int
    append: Optional[bool] = False
    lastChunk: Optional[bool] = False


class TaskStatus(BaseModel):
    state: TaskState
    timestamp: str
    message: Message


class Task(BaseModel):
    id: str
    sessionId: Optional[str] = None
    status: TaskStatus
    history: Optional[List[Message]] = []
    artifacts: Optional[List[Artifact]] = []
    metadata: Optional[Dict[str, object]] = {}


class TaskParams(BaseModel):
    id: str
    sessionId: Optional[str] = None
    message: Message


class PushNotificationConfig(BaseModel):
    url: str
    token: Optional[str] = None
    authentication: Optional[Dict[str, object]] = None


class TaskPushNotificationConfig(BaseModel):
    id: str
    pushNotificationConfig: PushNotificationConfig


class Result(BaseModel):
    id: str
    session_id: str
    status: TaskStatus

class RPCMethod(Enum):
    TASK_SEND = "tasks/send"
    TASK_GET = "tasks/get"
    TASK_CANCEL = "tasks/cancel"
    TASK_PUSH = "tasks/push"
    TASK_RESUBSCRIBE = "tasks/resubscribe"
    TASK_PUSH_GET = "tasks/push/get"
    TASK_PUSH_UPDATE = "tasks/push/update"
    TASK_PUSH_DELETE = "tasks/push/delete"

class RPCRequest(BaseModel):
    jsonrpc: str
    id: str
    method: RPCMethod
    params: TaskParams
    acceptedOutputModes: Optional[List[str]] = None
    pushNotification: Optional[Dict[str, object]] = None
    historyLength: Optional[int] = None
    metadata: Optional[Dict[str, object]] = {}

class RPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = Field(default_factory=lambda: uuid4().hex)
    result: Result | None = None
    error: JSONRPCError | None = None
