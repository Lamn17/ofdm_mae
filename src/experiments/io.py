import runtime as _runtime
import csv
from pathlib import Path

def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

class StreamingRows:
    """Write frame records immediately; paper grids must not accumulate millions of dicts."""
    def __init__(self, path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file=self.path.open('w',newline='')
        self.writer=None
        self.count=0

    def append(self,row):
        if self.writer is None:
            self.writer=csv.DictWriter(self.file,fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.count+=1
        if self.count%1000==0: self.file.flush()

    def finish(self):
        self.file.close()
        return {'path':str(self.path),'rows':self.count}
