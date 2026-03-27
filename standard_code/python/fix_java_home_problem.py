import os
import jdk4py
import scyjava


def fix_java_home_problem():
    # Point JAVA_HOME at jdk4py's bundled OpenJDK 21 (must happen before JVM starts).
    
    os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
    # Prevent scyjava from overriding JAVA_HOME with its own (broken) JVM finder.
    scyjava.config.set_java_constraints(fetch="never")

    # Workaround for jgo 2.1.2 bug on Windows: relativePath in a POM is relative to
    # the POM's *directory*, not the POM file path itself. Without this, jgo tries
    # to parse a directory as XML and raises PermissionError.
    from pathlib import Path
    import jgo.maven._core as _jgo_core
    def _fixed_pom_parent(self, pom):
        from jgo.maven._pom import POM as _POM
        if pom.element("parent") is None:
            return None
        g = pom.value("parent/groupId")
        a = pom.value("parent/artifactId")
        v = pom.value("parent/version")
        assert g and a and v
        relativePath = pom.value("parent/relativePath")
        if (
            isinstance(pom.source, Path)
            and relativePath
            and (parent_path := pom.source.parent / relativePath).exists()
            and parent_path.is_file()
        ):
            parent_pom = _POM(parent_path)
            if g == parent_pom.groupId and a == parent_pom.artifactId and v == parent_pom.version:
                return parent_pom
        pom_artifact = self.project(g, a).at_version(v).artifact(packaging="pom")
        return _POM(pom_artifact.resolve())
    _jgo_core.MavenContext.pom_parent = _fixed_pom_parent

# Should be run once
fix_java_home_problem()
