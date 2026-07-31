"""Qt-neutral controller for the SBT Project Console."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from SpatialBiologyToolkit.config.editing import ConfigFieldSpec
from SpatialBiologyToolkit.pipeline.inspection import (
    ExecutionInspection,
    ProjectOpenResult,
    ProjectSnapshot,
    context_with_config,
    inspect_execution,
    inspect_project,
    inspect_readiness,
    open_project_console,
    stage_documentation,
)
from SpatialBiologyToolkit.pipeline.models import (
    DependencyPolicy,
    ExecutionSummary,
    ModeSpec,
    ProjectRegistry,
    RegisteredProject,
    RunPlan,
    StageSpec,
)
from SpatialBiologyToolkit.pipeline.notes import ProjectNotesSession
from SpatialBiologyToolkit.pipeline.project_registry import (
    RegisteredProjectStatus,
    load_project_registry,
    register_project,
    registered_project_statuses,
    set_default_project,
    unregister_project,
)
from SpatialBiologyToolkit.pipeline.registry import MODES, STAGES, get_mode, get_stage


@dataclass
class ProjectConsoleController:
    """Own project sessions and expose only approved lightweight operations."""

    opened: ProjectOpenResult
    read_only: bool = False
    snapshot: ProjectSnapshot | None = None
    notes: ProjectNotesSession | None = None

    @classmethod
    def open(
        cls,
        project: str | Path | None,
        *,
        read_only: bool = False,
    ) -> "ProjectConsoleController":
        opened = open_project_console(project)
        controller = cls(opened=opened, read_only=read_only)
        if opened.context is not None:
            controller.refresh()
            controller.notes = ProjectNotesSession.open(opened.context)
        return controller

    @property
    def context(self):
        return self.opened.context

    @property
    def editor(self):
        return self.opened.editor

    @property
    def recovery(self):
        return self.opened.recovery

    @property
    def recovery_mode(self) -> bool:
        return self.opened.recovery_mode

    def effective_context(self):
        if self.context is None:
            raise RuntimeError("A valid project config is required for this view.")
        if self.editor is None:
            return self.context
        return context_with_config(self.context, self.editor.validated_config)

    def refresh(self) -> ProjectSnapshot:
        self.snapshot = inspect_project(self.effective_context())
        return self.snapshot

    def reload(self) -> None:
        root = self.opened.root
        existing_notes = self.notes
        self.opened = open_project_console(root)
        if self.opened.context is not None:
            self.refresh()
            self.notes = existing_notes or ProjectNotesSession.open(self.opened.context)

    def stages(self) -> tuple[StageSpec, ...]:
        return tuple(sorted(STAGES, key=lambda item: (item.catalogue_order, item.name)))

    def modes(self) -> tuple[ModeSpec, ...]:
        return tuple(MODES)

    def explain_stage(self, name: str) -> tuple[StageSpec, str]:
        stage = get_stage(name)
        return stage, stage_documentation(stage)

    def explain_mode(self, name: str) -> ModeSpec:
        return get_mode(name)

    def config_fields(self) -> list[ConfigFieldSpec]:
        if self.editor is None:
            return []
        return self.editor.field_specs()

    def config_sections_for_scope(self, kind: str, name: str) -> set[str]:
        """Return config sections linked to one stage or workflow mode."""

        if kind == "stage":
            return set(get_stage(name).config_sections)
        if kind == "mode":
            sections: set[str] = set()
            for stage_name in get_mode(name).stages:
                sections.update(get_stage(stage_name).config_sections)
            return sections
        return set()

    def readiness(
        self,
        target: str,
        *,
        dependency_policy: DependencyPolicy = "assets",
    ) -> RunPlan:
        return inspect_readiness(
            self.effective_context(),
            [target],
            dependency_policy=dependency_policy,
        )

    def project_registry(self) -> ProjectRegistry:
        return load_project_registry()

    def registered_projects(self) -> list[RegisteredProjectStatus]:
        return registered_project_statuses(self.project_registry())

    def register_current(
        self,
        *,
        name: str | None = None,
        make_default: bool = False,
    ) -> RegisteredProject:
        _registry, registered = register_project(
            self.opened.root,
            name=name,
            make_default=make_default,
        )
        return registered

    def register_path(
        self,
        project: str | Path,
        *,
        name: str | None = None,
        make_default: bool = False,
    ) -> RegisteredProject:
        _registry, registered = register_project(
            project,
            name=name,
            make_default=make_default,
        )
        return registered

    def unregister(self, reference: str | Path) -> RegisteredProject:
        _registry, removed = unregister_project(reference)
        return removed

    def set_default(self, reference: str | Path) -> RegisteredProject:
        _registry, selected = set_default_project(reference)
        return selected

    def executions(self) -> tuple[ExecutionSummary, ...]:
        return self.snapshot.executions if self.snapshot else ()

    def execution_detail(self, technical_run_id: str) -> ExecutionInspection:
        return inspect_execution(self.effective_context(), technical_run_id)


__all__ = ["ProjectConsoleController"]
