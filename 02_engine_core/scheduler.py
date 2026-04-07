from request import Request
from request import RequestStatus

class Scheduler:
    def __init__(self):
        self.queue = []
        self.step = 0
    
    def submit_request(self, request: Request):
        self.queue.append(request)
    
    def _get_active_requests(self):
        return [req for req in self.queue if req.request_status != RequestStatus.FINISHED and req.request_status != RequestStatus.ABORTED]
    
    def batch_decode(self):
        active_reqs = self._get_active_requests()
        if not active_reqs:
            return
        
        
        
    def run(self):
        print("Starting scheduler loop...")

        while True:
            active_reqs = self._get_active_requests()
            if not active_reqs:
                break
