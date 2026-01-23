from rich.console import Console
from rich.table import Table

from .reports import StabilityReport


class VerificationRender(Table):
    def __init__(self, report: StabilityReport):
        super().__init__(title=report.method + " Report")
        self.add_column("Check")
        self.add_column("Result")
        self.add_column("Details")
        self.add_row("Overall Stability", str(report.is_stable), report.message)
        
        for rep in report.details.values():
            if isinstance(rep, StabilityReport):
                self.add_row(rep.method, str(rep.is_stable), rep.message)

    def render(self):
        console = Console()
        console.print(self)
    