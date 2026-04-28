`global_lb` ist in deiner aktuellen Nutzung kein Flächenmaß, sondern der schlechteste noch offene Margin-Wert des gerade gelösten Gesamtproblems. In deinem Wrapper ziehst du ihn bereits aus `res.stats["bab"]` in abcrown_wrapper.py bis abcrown_wrapper.py. Wenn du 8 Root-Regionen zusammen in einem Solve prüfst, dann dominiert genau eine harte Restdomäne diesen Wert. Deshalb kann `global_lb` dir dort nicht sagen, wie viel Volumen schon “eigentlich sicher” ist. Er sagt nur: das gesamte verifizierte OR-von-AND-Problem ist noch mindestens so weit von `safe` entfernt.

**Was du aus ABCrown real ziehen kannst**
Du bekommst praktisch nur diese stabilen Dinge zurück:

- `status` und `success`
- `stats["bab"][-1] = [index, global_lb, visited_domains, bab_time]`
- `stats["attack_examples"]`, `stats["all_adv_candidates"]`, `stats["attack_margins"]`
- `reference`, aber das ist oft leer oder enthält nur interne Incomplete-Verifier-Artefakte, keine Liste zertifizierter Teilboxen

Das passt auch zu deinem aktuellen Code in abcrown_wrapper.py und abcrown_wrapper.py: du reduzierst den Solver am Ende auf ein Bool. Wenn du mehr nutzen willst, musst du `_solve_box_with_model` und `_solve_root_regions_batched` so ändern, dass sie statt Bool ein kleines Ergebnisobjekt zurückgeben, zum Beispiel mit `status`, `global_lb`, `visited_domains`, `witness_reproducible`.

**Wie du `global_lb` sinnvoll nutzen kannst**
Nicht als Zertifikat, aber als Prioritätssignal.

Am sinnvollsten wird `global_lb`, wenn du nicht 8 Boxen gemeinsam, sondern jede Box separat löst. Dann ist es ein guter Härte-Score pro Box:

- `global_lb` deutlich über Schwellwert: Box ist sicher, nicht weiter splitten
- `global_lb` knapp unter Schwellwert: gute Kandidaten zum Splitten, hier lohnt weitere Arbeit
- `global_lb` sehr stark negativ und Witness reproduzierbar: vermutlich echte unsichere Box
- `global_lb` sehr stark negativ und Witness nicht reproduzierbar: meist nur sehr lose Bounds, also eher unresolved als unsafe

Für eine adaptive Suche würde ich unresolved-Boxen nach einer Priorität wie “großes Volumen und `global_lb` nahe an der Entscheidungsschwelle” behandeln. Große Boxen mit extrem negativem `global_lb` sind oft schlechte ROI-Kandidaten, wenn du primär zertifiziertes Volumen maximieren willst.

**Wie eine volumenbasierte partielle Zertifizierung aussehen würde**
Das würde ich explizit außerhalb von ABCrown modellieren, nicht vom Solver erwarten.

Für jede Box \(B_i = [\ell_i, u_i]\) führst du einen Zustand:

- `outside`: sicher außerhalb von \(V \le \rho\)
- `certified`: sicher innerhalb und Lyapunov-Bedingung bewiesen
- `unsafe`: echter reproduzierbarer Verstoß
- `unresolved`: Timeout, `unknown` oder zu lose Bounds

Dann rechnest du Volumina:

- \( \mathrm{vol}(B_i) = \prod_d (u_{i,d} - \ell_{i,d}) \)
- zertifiziertes Volumen = Summe aller `certified`-Boxen
- unresolved-Volumen = Summe aller `unresolved`-Boxen
- outside-Volumen = Summe aller `outside`-Boxen

Der entscheidende Trick ist: Nutze LiRPA nicht als schnellen Voll-Certifier für die gesamte Bedingung, sondern nur für Skalar-Bounds auf \(V\). Das ist viel leichter.

Für jede Box berechnest du \( [\underline V_i, \overline V_i] \):

- wenn \( \underline V_i > \rho \): Box ist `outside`
- wenn \( \overline V_i \le \rho \): Box liegt vollständig innerhalb der Sublevel-Menge
- wenn \( \underline V_i \le \rho < \overline V_i \): Box ist eine Boundary-Box

Das ist viel wertvoller als der schnelle volle LiRPA-Precertifier, der bei dir nichts zertifiziert hat. Auf \(V\) allein sollte LiRPA deutlich informativer sein.

Dann:

- Für `inside`-Boxen prüfst du nur noch die eigentliche Bedingung für Dynamik und Decrease.
- Für Boundary-Boxen splittest du erst einmal weiter.
- ABCrown läuft nur noch auf `inside`- oder kleinen Boundary-Boxen, nicht auf allem.

Das würde auch eine saubere Erweiterung von certifier_base.py und certifier_base.py nahelegen: Neben `certified`, `failed`, `outside_sublevel` brauchst du eigentlich auch `unresolved`.

**Kann man das mit rho-Suche kombinieren?**
Ja, und sogar effizienter als dein aktueller Pfad.

Der wichtige Punkt ist: Die \(V\)-Intervalle einer Box sind rho-unabhängig. Du kannst sie cachen.

Für jede Box hast du dann feste Schwellen:

- für \( \rho < \underline V_i \): Box ist sicher `outside`
- für \( \rho \ge \overline V_i \): Box ist vollständig im Sublevel
- nur beim Übergang dazwischen ist sie Boundary

Das bedeutet:

- Bei einer neuen rho-Auswertung musst du nicht alles neu rechnen.
- Du musst nur Boxen neu anschauen, deren \(V\)-Intervall die neue rho schneidet oder die gerade von Boundary zu fully-inside wechseln.
- Eine bereits bewiesene Condition-Zertifizierung einer fully-inside-Box kannst du für größere rho direkt wiederverwenden.

Das ist viel günstiger als dein aktuelles “für jedes rho wieder global 8 große Root-Boxen lösen”.

**Wie ich es praktisch aufbauen würde**
1. Externer Box-Cache: `lb`, `ub`, `volume`, `V_lb`, `V_ub`, `condition_status`, `last_global_lb`.
2. Günstiger Vorfilter auf \(V\): outside / inside / boundary.
3. ABCrown nur noch auf inside-Boxen oder kleinen Boundary-Boxen.
4. `unknown` nie direkt als `failed` zählen, sondern als `unresolved`.
5. Best-first-Splitting nach Volumen und `global_lb`-Nähe zur Schwelle.
6. Abbruch nach Zeitbudget oder unresolved-Volumen unter Toleranz.

**Welche rho-Suche dazu passt**
Wenn du wirklich “max rho auf maximaler Fläche” willst, hast du eigentlich ein Zwei-Ziel-Problem. Dann brauchst du eine explizite Zieldefinition.

Sinnvolle Varianten sind:

1. Maximiere rho unter der Nebenbedingung: zertifiziertes Volumen \(\ge V_{\min}\)
2. Maximiere rho unter der Nebenbedingung: unresolved-Volumen \(\le \varepsilon\)
3. Berechne eine Pareto-Kurve aus \((\rho, \text{zertifiziertes Volumen})\)

Die klassische Bisection passt nur sauber zu einem einzigen monotonen Kriterium. Für “zertifiziertes Volumen mindestens \(V_{\min}\)” kann das funktionieren. Für “zertifizierter Anteil” meist nicht, weil der Anteil nicht sauber monoton sein muss.

**Was ich konkret empfehlen würde**
Die beste nächste Ausbaustufe ist nicht, noch mehr aus dem internen ABCrown-Input-Splitting herauszuquetschen. Die API gibt dir dafür zu wenig. Stattdessen:

1. Nutze LiRPA auf \(V\) allein für Inside/Outside/Boundary-Klassifikation.
2. Führe `unresolved` als eigene Regionklasse ein.
3. Lass ABCrown pro Box einen Rich-Result statt Bool zurückgeben.
4. Baue darauf eine budgetierte partielle Volumen-Zertifizierung mit Cache über rho.

Wenn du willst, kann ich das als nächsten Schritt konkret in deinem Code designen und dann stufenweise implementieren:
1. `SolveResult`-Wrapper pro Box
2. `unresolved_regions` in den Result-Dataclasses
3. \(V\)-Intervall-Filter zusätzlich zum aktuellen Outside-Filter
4. budgetierte partielle rho-Auswertung mit Cache