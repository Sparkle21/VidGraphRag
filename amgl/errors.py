"""异常不携带候选原文，避免通过 HTTP 或日志泄漏敏感内容。"""

class AMGLError(Exception):
    code = 40001
    status = 400


class NotFound(AMGLError):
    code = 40401
    status = 404


class Conflict(AMGLError):
    code = 40901
    status = 409


class QueueFull(AMGLError):
    code = 42901
    status = 429


class NotReady(AMGLError):
    code = 50302
    status = 503


class UnsafeResume(Conflict):
    code = 40902
