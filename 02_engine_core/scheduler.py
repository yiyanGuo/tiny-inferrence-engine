from request import Request
from request import RequestStatus

class Scheduler:
    def __init__(self, max_prefill_batch_size=8, max_decode_batch_size=32):
        self.requests: dict[str, Request] = {}  # {request_id: Request}
        self.step = 0
        self.max_prefill_batch_size = max_prefill_batch_size
        self.max_decode_batch_size = max_decode_batch_size

    def submit_request(self, request: Request):
        self.requests[request.request_id] = request
    
    def schedule(self):
        prefilling = [req for req in self.requests.values() if req.request_status == RequestStatus.PREFILLING][:self.max_prefill_batch_size]
        decoding = [req for req in self.requests.values() if req.request_status == RequestStatus.DECODING][:self.max_decode_batch_size]
        return prefilling, decoding
    
    def update_after_step(self, finished_request_ids: list[str]):
        # This method can be implemented to update the scheduler state after each step
        for req_id in finished_request_ids:
            self.requests.pop(req_id, None)
    
    def has_active_requests(self):
        return any(req.request_status in [RequestStatus.PREFILLING, RequestStatus.DECODING] for req in self.requests.values())