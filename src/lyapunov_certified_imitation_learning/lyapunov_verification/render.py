from rich.console import Console
from rich.table import Table

from .stability_report import StabilityReport


class ResultsTable(Table):
    def __init__(self, title: str):
        super().__init__(title=title)
        self.add_column("Check")
        self.add_column("Result")
        self.add_column("Details")
        
    def render(self):
        console = Console()
        console.print(self)


class EmpiricalVerificationRender(ResultsTable):
    def __init__(self, report: StabilityReport):
        super().__init__(title="Empirical Verification")
        self.add_row("Overall Stability", str(report.is_stable), report.message)
        
        for rep in report.details.values():
            if isinstance(rep, StabilityReport):
                self.add_row(rep.method, str(rep.is_stable), rep.message)

class FormalVerificationRender(ResultsTable):
    def __init__(self, report: StabilityReport):
        super().__init__(title="Formal Verification")
        self.add_row("Overall Stability", str(report.is_stable), report.message)
        
        for rep in report.details.values():
            if isinstance(rep, StabilityReport):
                self.add_row(rep.method, str(rep.is_stable), rep.message)
        
    