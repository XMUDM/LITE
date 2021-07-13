/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import java.io._
import java.nio.file.Files

import scala.io.Source
import scala.util.Properties
import scala.collection.JavaConverters._
import scala.collection.mutable.Stack

import sbt._
import sbt.Classpaths.publishTask
import sbt.Keys._
import sbtunidoc.Plugin.UnidocKeys.unidocGenjavadocVersion
import com.etsy.sbt.checkstyle.CheckstylePlugin.autoImport._
import com.simplytyped.Antlr4Plugin._
import com.typesafe.sbt.pom.{PomBuild, SbtPomKeys}
import com.typesafe.tools.mima.plugin.MimaKeys
import org.scalastyle.sbt.ScalastylePlugin.autoImport._
import org.scalastyle.sbt.Tasks

import spray.revolver.RevolverPlugin._

object BuildCommons {

  private val buildLocation = file(".").getAbsoluteFile.getParentFile

  val sqlProjects@Seq(catalyst, sql, hive, hiveThriftServer, sqlKafka010, avro) = Seq(
    "catalyst", "sql", "hive", "hive-thriftserver", "sql-kafka-0-10", "avro"
  ).map(ProjectRef(buildLocation, _))

  val streamingProjects@Seq(streaming, streamingKafka010) =
    Seq("streaming", "streaming-kafka-0-10").map(ProjectRef(buildLocation, _))

  val allProjects@Seq(
    core, graphx, mllib, mllibLocal, repl, networkCommon, networkShuffle, launcher, unsafe, tags, sketch, kvstore, _*
  ) = Seq(
    "core", "graphx", "mllib", "mllib-local", "repl", "network-common", "network-shuffle", "launcher", "unsafe",
    "tags", "sketch", "kvstore"
  ).map(ProjectRef(buildLocation, _)) ++ sqlProjects ++ streamingProjects

  val optionallyEnabledProjects@Seq(kubernetes, mesos, yarn,
    streamingFlumeSink, streamingFlume,
    streamingKafka, sparkGangliaLgpl, streamingKinesisAsl,
    dockerIntegrationTests, hadoopCloud, kubernetesIntegrationTests) =
    Seq("kubernetes", "mesos", "yarn",
      "streaming-flume-sink", "streaming-flume",
      "streaming-kafka-0-8", "ganglia-lgpl", "streaming-kinesis-asl",
      "docker-integration-tests", "hadoop-cloud", "kubernetes-integration-tests").map(ProjectRef(buildLocation, _))

  val assemblyProjects@Seq(networkYarn, streamingFlumeAssembly, streamingKafkaAssembly, streamingKafka010Assembly, streamingKinesisAslAssembly) =
    Seq("network-yarn", "streaming-flume-assembly", "streaming-kafka-0-8-assembly", "streaming-kafka-0-10-assembly", "streaming-kinesis-asl-assembly")
      .map(ProjectRef(buildLocation, _))

  val copyJarsProjects@Seq(assembly, examples) = Seq("assembly", "examples")
    .map(ProjectRef(buildLocation, _))

  val tools = ProjectRef(buildLocation, "tools")
  // Root project.
  val spark = ProjectRef(buildLocation, "spark")
  val sparkHome = buildLocation

  val testTempDir = s"$sparkHome/target/tmp"

  val javacJVMVersion = settingKey[String]("source and target JVM version for javac")
  val scalacJVMVersion = settingKey[String]("source and target JVM version for scalac")
}

object SparkBuild extends PomBuild {

  import BuildCommons._
  import scala.collection.mutable.Map

  val projectsMap: Map[String, Seq[Setting[_]]] = Map.empty

  override val profiles = {
    val profiles = Properties.envOrNone("SBT_MAVEN_PROFILES") match {
      case None => Seq("sbt")
      case Some(v) =>
        v.split("(\\s+|,)").filterNot(_.isEmpty).map(_.trim.replaceAll("-P", "")).toSeq
    }

    Option(System.getProperty("scala.version"))
      .filter(_.startsWith("2.12"))
      .foreach { versionString =>
        System.setProperty("scala-2.12", "true")
      }
    if (System.getProperty("scala-2.12") == "") {
      // To activate scala-2.10 profile, replace empty property value to non-empty value
      // in the same way as Maven which handles -Dname as -Dname=true before executes build process.
      // see: https://github.com/apache/maven/blob/maven-3.0.4/maven-embedder/src/main/java/org/apache/maven/cli/MavenCli.java#L1082
      System.setProperty("scala-2.12", "true")
    }
    profiles
  }

  Properties.envOrNone("SBT_MAVEN_PROPERTIES") match {
    case Some(v) =>
      v.split("(\\s+|,)").filterNot(_.isEmpty).map(_.split("=")).foreach(x => System.setProperty(x(0), x(1)))
    case _ =>
  }

  override val userPropertiesMap = System.getProperties.asScala.toMap

  lazy val MavenCompile = config("m2r") extend(Compile)
  lazy val publishLocalBoth = TaskKey[Unit]("publish-local", "publish local for m2 and ivy")

  lazy val sparkGenjavadocSettings: Seq[sbt.Def.Setting[_]] = Seq(
    libraryDependencies += compilerPlugin(
      "com.typesafe.genjavadoc" %% "genjavadoc-plugin" % unidocGenjavadocVersion.value cross CrossVersion.full),
    scalacOptions ++= Seq(
      "-P:genjavadoc:out=" + (target.value / "java"),
      "-P:genjavadoc:strictVisibility=true" // hide package private types
    )
  )

  lazy val scalaStyleRules = Project("scalaStyleRules", file("scalastyle"))
    .settings(
      libraryDependencies += "org.scalastyle" %% "scalastyle" % "1.0.0"
    )

  lazy val scalaStyleOnCompile = taskKey[Unit]("scalaStyleOnCompile")

  lazy val scalaStyleOnTest = taskKey[Unit]("scalaStyleOnTest")

  // We special case the 'println' lint rule to only be a warning on compile, because adding
  // printlns for debugging is a common use case and is easy to remember to remove.
  val scalaStyleOnCompileConfig: String = {
    val in = "scalastyle-config.xml"
    val out = "scalastyle-on-compile.generated.xml"
    val replacements = Map(
      """customId="println" level="error"""" -> """customId="println" level="warn""""
    )
    var contents = Source.fromFile(in).getLines.mkString("\n")
    for ((k, v) <- replacements) {
      require(contents.contains(k), s"Could not rewrite '$k' in original scalastyle config.")
      contents = contents.replace(k, v)
    }
    new PrintWriter(out) {
      write(contents)
      close()
    }
    out
  }

  // Return a cached scalastyle task for a given configuration (usually Compile or Test)
  private def cachedScalaStyle(config: Configuration) = Def.task {
    val logger = streams.value.log
    // We need a different cache dir per Configuration, otherwise they collide
    val cacheDir = target.value / s"scalastyle-cache-${config.name}"
    val cachedFun = FileFunction.cached(cacheDir, FilesInfo.lastModified, FilesInfo.exists) {
      (inFiles: Set[File]) => {
        val args: Seq[String] = Seq.empty
        val scalaSourceV = Seq(file(scalaSource.in(config).value.getAbsolutePath))
        val configV = (baseDirectory in ThisBuild).value / scalaStyleOnCompileConfig
        val configUrlV = scalastyleConfigUrl.in(config).value
        val streamsV = streams.in(config).value
        val failOnErrorV = true
        val failOnWarningV = false
        val scalastyleTargetV = scalastyleTarget.in(config).value
        val configRefreshHoursV = scalastyleConfigRefreshHours.in(config).value
        val targetV = target.in(config).value
        val configCacheFileV = scalastyleConfigUrlCacheFile.in(config).value

        logger.info(s"Running scalastyle on ${name.value} in ${config.name}")
        Tasks.doScalastyle(args, configV, configUrlV, failOnErrorV, failOnWarningV, scalaSourceV,
          scalastyleTargetV, streamsV, configRefreshHoursV, targetV, configCacheFileV)

        Set.empty
      }
    }

    cachedFun(findFiles(scalaSource.in(config).value))
  }

  private def findFiles(file: File): Set[File] = if (file.isDirectory) {
    file.listFiles().toSet.flatMap(findFiles) + file
  } else {
    Set(file)
  }

  def enableScalaStyle: Seq[sbt.Def.Setting[_]] = Seq(
    scalaStyleOnCompile := cachedScalaStyle(Compile).value,
    scalaStyleOnTest := cachedScalaStyle(Test).value,
    logLevel in scalaStyleOnCompile := Level.Warn,
    logLevel in scalaStyleOnTest := Level.Warn,
    (compile in Compile) := {
      scalaStyleOnCompile.value
      (compile in Compile).value
    },
    (compile in Test) := {
      scalaStyleOnTest.value
      (compile in Test).value
    }
  )

  lazy val sharedSettings = sparkGenjavadocSettings ++
      (if (sys.env.contains("NOLINT_ON_COMPILE")) Nil else enableScalaStyle) ++ Seq(
    exportJars in Compile := true,
    exportJars in Test := false,
    javaHome := sys.env.get("JAVA_HOME")
      .orElse(sys.props.get("java.home").map { p => new File(p).getParentFile().getAbsolutePath() })
      .map(file),
    incOptions := incOptions.value.withNameHashing(true),
    publishMavenStyle := true,
    unidocGenjavadocVersion := "0.14",

    // Override SBT's default resolvers:
    resolvers := Seq(
      // Google Mirror of Maven Central, placed first so that it's used instead of flaky Maven Central.
      // See https://storage-download.googleapis.com/maven-central/index.html for more info.
      "gcs-maven-central-mirror" at "https://maven-central.storage-download.googleapis.com/maven2/",
      DefaultMavenRepository,
      Resolver.mavenLocal,
      Resolver.file("local", file(Path.userHome.absolutePath + "/.ivy2/local"))(Resolver.ivyStylePatterns)
    ),
    externalResolvers := resolvers.value,
    otherResolvers := SbtPomKeys.mvnLocalRepository(dotM2 => Seq(Resolver.file("dotM2", dotM2))).value,
    publishLocalConfiguration in MavenCompile :=
      new PublishConfiguration(None, "dotM2", packagedArtifacts.value, Seq(), ivyLoggingLevel.value),
    publishMavenStyle in MavenCompile := true,
    publishLocal in MavenCompile := publishTask(publishLocalConfiguration in MavenCompile, deliverLocal).value,
    publishLocalBoth := Seq(publishLocal in MavenCompile, publishLocal).dependOn.value,

    javacOptions in (Compile, doc) ++= {
      val versionParts = System.getProperty("java.version").split("[+.\\-]+", 3)
      var major = versionParts(0).toInt
      if (major == 1) major = versionParts(1).toInt
      if (major >= 8) Seq("-Xdoclint:all", "-Xdoclint:-missing") else Seq.empty
    },

    javacJVMVersion := "1.8",
    scalacJVMVersion := "1.8",

    javacOptions in Compile ++= Seq(
      "-encoding", "UTF-8",
      "-source", javacJVMVersion.value
    ),
    // This -target and Xlint:unchecked options cannot be set in the Compile configuration scope since
    // `javadoc` doesn't play nicely with them; see https://github.com/sbt/sbt/issues/355#issuecomment-3817629
    // for additional discussion and explanation.
    javacOptions in (Compile, compile) ++= Seq(
      "-target", javacJVMVersion.value,
      "-Xlint:unchecked"
    ),

    scalacOptions in Compile ++= Seq(
      s"-target:jvm-${scalacJVMVersion.value}",
      "-sourcepath", (baseDirectory in ThisBuild).value.getAbsolutePath  // Required for relative source links in scaladoc
    ),

    // Remove certain packages from Scaladoc
    scalacOptions in (Compile, doc) := Seq(
      "-groups",
      "-skip-packages", Seq(
        "org.apache.spark.api.python",
        "org.apache.spark.network",
        "org.apache.spark.deploy",
        "org.apache.spark.util.collection"
      ).mkString(":"),
      "-doc-title", "Spark " + version.value.replaceAll("-SNAPSHOT", "") + " ScalaDoc"
    ) ++ {
      // Do not attempt to scaladoc javadoc comments under 2.12 since it can't handle inner classes
      if (scalaBinaryVersion.value == "2.12") Seq("-no-java-comments") else Seq.empty
    },

    // Implements -Xfatal-warnings, ignoring deprecation warnings.
    // Code snippet taken from https://issues.scala-lang.org/browse/SI-8410.
    compile in Compile := {
      val analysis = (compile in Compile).value
      val out = streams.value

      def logProblem(l: (=> String) => Unit, f: File, p: xsbti.Problem) = {
        l(f.toString + ":" + p.position.line.fold("")(_ + ":") + " " + p.message)
        l(p.position.lineContent)
        l("")
      }

      var failed = 0
      analysis.infos.allInfos.foreach { case (k, i) =>
        i.reportedProblems foreach { p =>
          val deprecation = p.message.contains("is deprecated")

          if (!deprecation) {
            failed = failed + 1
          }

          val printer: (=> String) => Unit = s => if (deprecation) {
            out.log.warn(s)
          } else {
            out.log.error("[warn] " + s)
          }

          logProblem(printer, k, p)

        }
      }

      if (failed > 0) {
        sys.error(s"$failed fatal warnings")
      }
      analysis
    }
  )

  def enable(settings: Seq[Setting[_]])(projectRef: ProjectRef) = {
    val existingSettings = projectsMap.getOrElse(projectRef.project, Seq[Setting[_]]())
    projectsMap += (projectRef.project -> (existingSettings ++ settings))
  }

  // Note ordering of these settings matter.
  /* Enable shared settings on all projects */
  (allProjects ++ optionallyEnabledProjects ++ assemblyProjects ++ copyJarsProjects ++ Seq(spark, tools))
    .foreach(enable(sharedSettings ++ DependencyOverrides.settings ++
      ExcludedDependencies.settings ++ Checkstyle.settings))

  /* Enable tests settings for all projects except examples, assembly and tools */
  (allProjects ++ optionallyEnabledProjects).foreach(enable(TestSettings.settings))

  val mimaProjects = allProjects.filterNot { x =>
    Seq(
      spark, hive, hiveThriftServer, catalyst, repl, networkCommon, networkShuffle, networkYarn,
      unsafe, tags, sqlKafka010, kvstore, avro
    ).contains(x)
  }

  mimaProjects.foreach { x =>
    enable(MimaBuild.mimaSettings(sparkHome, x))(x)
  }

  /* Generate and pick the spark build info from extra-resources */
  enable(Core.settings)(core)

  /* Unsafe settings */
  enable(Unsafe.settings)(unsafe)

  /*
   * Set up tasks to copy dependencies during packaging. This step can be disabled in the command
   * line, so that dev/mima can run without trying to copy these files again and potentially
   * causing issues.
   */
  if (!"false".equals(System.getProperty("copyDependencies"))) {
    copyJarsProjects.foreach(enable(CopyDependencies.settings))
  }

  /* Enable Assembly for all assembly projects */
  assemblyProjects.foreach(enable(Assembly.settings))

  /* Package pyspark artifacts in a separate zip file for YARN. */
  enable(PySparkAssembly.settings)(assembly)

  /* Enable unidoc only for the root spark project */
  enable(Unidoc.settings)(spark)

  /* Catalyst ANTLR generation settings */
  enable(Catalyst.settings)(catalyst)

  /* Spark SQL Core console settings */
  enable(SQL.settings)(sql)

  /* Hive console settings */
  enable(Hive.settings)(hive)

  enable(Flume.settings)(streamingFlumeSink)

  // SPARK-14738 - Remove docker tests from main Spark build
  // enable(DockerIntegrationTests.settings)(dockerIntegrationTests)

  /**
   * Adds the ability to run the spark shell directly from SBT without building an assembly
   * jar.
   *
   * Usage: `build/sbt sparkShell`
   */
  val sparkShell = taskKey[Unit]("start a spark-shell.")
  val sparkPackage = inputKey[Unit](
    s"""
       |Download and run a spark package.
       |Usage `builds/sbt "sparkPackage <group:artifact:version> <MainClass> [args]
     """.stripMargin)
  val sparkSql = taskKey[Unit]("starts the spark sql CLI.")

  enable(Seq(
    connectInput in run := true,
    fork := true,
    outputStrategy in run := Some (StdoutOutput),

    javaOptions += "-Xmx2g",

    sparkShell := {
      (runMain in Compile).toTask(" org.apache.spark.repl.Main -usejavacp").value
    },

    sparkPackage := {
      import complete.DefaultParsers._
      val packages :: className :: otherArgs = spaceDelimited("<group:artifact:version> <MainClass> [args]").parsed.toList
      val scalaRun = (runner in run).value
      val classpath = (fullClasspath in Runtime).value
      val args = Seq("--packages", packages, "--class", className, (Keys.`package` in Compile in LocalProject("core"))
        .value.getCanonicalPath) ++ otherArgs
      println(args)
      scalaRun.run("org.apache.spark.deploy.SparkSubmit", classpath.map(_.data), args, streams.value.log)
    },

    javaOptions in Compile += "-Dspark.master=local",

    sparkSql := {
      (runMain in Compile).toTask(" org.apache.spark.sql.hive.thriftserver.SparkSQLCLIDriver").value
    }
  ))(assembly)

  enable(Seq(sparkShell := sparkShell in LocalProject("assembly")))(spark)

  // TODO: move this to its upstream project.
  override def projectDefinitions(baseDirectory: File): Seq[Project] = {
    super.projectDefinitions(baseDirectory).map { x =>
      if (projectsMap.exists(_._1 == x.id)) x.settings(projectsMap(x.id): _*)
      else x.settings(Seq[Setting[_]](): _*)
    } ++ Seq[Project](OldDeps.project)
  }

  if (!sys.env.contains("SERIAL_SBT_TESTS")) {
    allProjects.foreach(enable(SparkParallelTestGrouping.settings))
  }
}

object SparkParallelTestGrouping {
  // Settings for parallelizing tests. The basic strategy here is to run the slowest suites (or
  // collections of suites) in their own forked JVMs, allowing us to gain parallelism within a
  // SBT project. Here, we take a whitelisting approach where the default behavior is to run all
  // tests sequentially in a single JVM, requiring us to manually opt-in to the extra parallelism.
  //
  // There are a reasons why such a whitelist approach is good:
  //
  //    1. Launching one JVM per suite adds significant overhead for short-running suites. In
  //       addition to JVM startup time and JIT warmup, it appears that initialization of Derby
  //       metastores can be very slow so creating a fresh warehouse per suite is inefficient.
  //
  //    2. When parallelizing within a project we need to give each forked JVM a different tmpdir
  //       so that the metastore warehouses do not collide. Unfortunately, it seems that there are
  //       some tests which have an overly tight dependency on the default tmpdir, so those fragile
  //       tests need to continue re-running in the default configuration (or need to be rewritten).
  //       Fixing that problem would be a huge amount of work for limited payoff in most cases
  //       because most test suites are short-running.
  //

  private val testsWhichShouldRunInTheirOwnDedicatedJvm = Set(
    "org.apache.spark.DistributedSuite",
    "org.apache.spark.sql.catalyst.expressions.DateExpressionsSuite",
    "org.apache.spark.sql.catalyst.expressions.HashExpressionsSuite",
    "org.apache.spark.sql.catalyst.expressions.CastSuite",
    "org.apache.spark.sql.catalyst.expressions.MathExpressionsSuite",
    "org.apache.spark.sql.hive.HiveExternalCatalogSuite",
    "org.apache.spark.sql.hive.StatisticsSuite",
    "org.apache.spark.sql.hive.execution.HiveCompatibilitySuite",
    "org.apache.spark.sql.hive.client.VersionsSuite",
    "org.apache.spark.sql.hive.client.HiveClientVersions",
    "org.apache.spark.sql.hive.HiveExternalCatalogVersionsSuite",
    "org.apache.spark.ml.classification.LogisticRegressionSuite",
    "org.apache.spark.ml.classification.LinearSVCSuite",
    "org.apache.spark.sql.SQLQueryTestSuite"
  )

  private val DEFAULT_TEST_GROUP = "default_test_group"

  private def testNameToTestGroup(name: String): String = name match {
    case _ if testsWhichShouldRunInTheirOwnDedicatedJvm.contains(name) => name
    case _ => DEFAULT_TEST_GROUP
  }

  lazy val settings = Seq(
    testGrouping in Test := {
      val tests: Seq[TestDefinition] = (definedTests in Test).value
      val defaultForkOptions = ForkOptions(
        bootJars = Nil,
        javaHome = javaHome.value,
        connectInput = connectInput.value,
        outputStrategy = outputStrategy.value,
        runJVMOptions = (javaOptions in Test).value,
        workingDirectory = Some(baseDirectory.value),
        envVars = (envVars in Test).value
      )
      tests.groupBy(test => testNameToTestGroup(test.name)).map { case (groupName, groupTests) =>
        val forkOptions = {
          if (groupName == DEFAULT_TEST_GROUP) {
            defaultForkOptions
          } else {
            defaultForkOptions.copy(runJVMOptions = defaultForkOptions.runJVMOptions ++
              Seq(s"-Djava.io.tmpdir=${baseDirectory.value}/target/tmp/$groupName"))
          }
        }
        new Tests.Group(
          name = groupName,
          tests = groupTests,
          runPolicy = Tests.SubProcess(forkOptions))
      }
    }.toSeq
  )
}

object Core {
  lazy val settings = Seq(
    resourceGenerators in Compile += Def.task {
      val buildScript = baseDirectory.value + "/../build/spark-build-info"
      val targetDir = baseDirectory.value + "/target/extra-resources/"
      val command = Seq("bash", buildScript, targetDir, version.value)
      Process(command).!!
      val propsFile = baseDirectory.value / "target" / "extra-resources" / "spark-version-info.properties"
      Seq(propsFile)
    }.taskValue
  )
}

object Unsafe {
  lazy val settings = Seq(
    // This option is needed to suppress warnings from sun.misc.Unsafe usage
    javacOptions in Compile += "-XDignore.symbol.file"
  )
}

object Flume {
  lazy val settings = sbtavro.SbtAvro.avroSettings
}

object DockerIntegrationTests {
  // This serves to override the override specified in DependencyOverrides:
  lazy val settings = Seq(
    dependencyOverrides += "com.google.guava" % "guava" % "18.0",
    resolvers += "DB2" at "https://app.camunda.com/nexus/content/repositories/public/",
    libraryDependencies += "com.oracle" % "ojdbc6" % "11.2.0.1.0" from "https://app.camunda.com/nexus/content/repositories/public/com/oracle/ojdbc6/11.2.0.1.0/ojdbc6-11.2.0.1.0.jar" // scalastyle:ignore
  )
}

/**
 * Overrides to work around sbt's dependency resolution being different from Maven's.
 */
object DependencyOverrides {
  lazy val settings = Seq(
    dependencyOverrides += "com.google.guava" % "guava" % "14.0.1",
    dependencyOverrides += "commons-io" % "commons-io" % "2.4",
    dependencyOverrides += "com.fasterxml.jackson.core"  % "jackson-databind" % "2.6.7.3",
    dependencyOverrides += "jline" % "jline" % "2.14.6")
}

/**
 * This excludes library dependencies in sbt, which are specified in maven but are
 * not needed by sbt build.
 */
object ExcludedDependencies {
  lazy val settings = Seq(
    libraryDependencies ~= { libs => libs.filterNot(_.name == "groovy-all") }
  )
}

/**
 * Project to pull previous artifacts of Spark for generating Mima excludes.
 */
object OldDeps {

  lazy val project = Project("oldDeps", file("dev"), settings = oldDepsSettings)

  lazy val allPreviousArtifactKeys = Def.settingDyn[Seq[Set[ModuleID]]] {
    SparkBuild.mimaProjects
      .map { project => MimaKeys.mimaPreviousArtifacts in project }
      .map(k => Def.setting(k.value))
      .join
  }

  def oldDepsSettings() = Defaults.coreDefaultSettings ++ Seq(
    name := "old-deps",
    libraryDependencies := allPreviousArtifactKeys.value.flatten
  )
}

object Catalyst {
  lazy val settings = antlr4Settings ++ Seq(
    antlr4Version in Antlr4 := "4.7",
    antlr4PackageName in Antlr4 := Some("org.apache.spark.sql.catalyst.parser"),
    antlr4GenListener in Antlr4 := true,
    antlr4GenVisitor in Antlr4 := true
  )
}

object SQL {
  lazy val settings = Seq(
    initialCommands in console :=
      """
        |import org.apache.spark.SparkContext
        |import org.apache.spark.sql.SQLContext
        |import org.apache.spark.sql.catalyst.analysis._
        |import org.apache.spark.sql.catalyst.dsl._
        |import org.apache.spark.sql.catalyst.errors._
        |import org.apache.spark.sql.catalyst.expressions._
        |import org.apache.spark.sql.catalyst.plans.logical._
        |import org.apache.spark.sql.catalyst.rules._
        |import org.apache.spark.sql.catalyst.util._
        |import org.apache.spark.sql.execution
        |import org.apache.spark.sql.functions._
        |import org.apache.spark.sql.types._
        |
        |val sc = new SparkContext("local[*]", "dev-shell")
        |val sqlContext = new SQLContext(sc)
        |import sqlContext.implicits._
        |import sqlContext._
      """.stripMargin,
    cleanupCommands in console := "sc.stop()"
  )
}

object Hive {

  lazy val settings = Seq(
    // Specially disable assertions since some Hive tests fail them
    javaOptions in Test := (javaOptions in Test).value.filterNot(_ == "-ea"),
    // Supporting all SerDes requires us to depend on deprecated APIs, so we turn off the warnings
    // only for this subproject.
    scalacOptions := (scalacOptions map { currentOpts: Seq[String] =>
      currentOpts.filterNot(_ == "-deprecation")
    }).value,
    initialCommands in console :=
      """
        |import org.apache.spark.SparkContext
        |import org.apache.spark.sql.catalyst.analysis._
        |import org.apache.spark.sql.catalyst.dsl._
        |import org.apache.spark.sql.catalyst.errors._
        |import org.apache.spark.sql.catalyst.expressions._
        |import org.apache.spark.sql.catalyst.plans.logical._
        |import org.apache.spark.sql.catalyst.rules._
        |import org.apache.spark.sql.catalyst.util._
        |import org.apache.spark.sql.execution
        |import org.apache.spark.sql.functions._
        |import org.apache.spark.sql.hive._
        |import org.apache.spark.sql.hive.test.TestHive._
        |import org.apache.spark.sql.hive.test.TestHive.implicits._
        |import org.apache.spark.sql.types._""".stripMargin,
    cleanupCommands in console := "sparkContext.stop()",
    // Some of our log4j jars make it impossible to submit jobs from this JVM to Hive Map/Reduce
    // in order to generate golden files.  This is only required for developers who are adding new
    // new query tests.
    fullClasspath in Test := (fullClasspath in Test).value.filterNot { f => f.toString.contains("jcl-over") }
  )
}

object Assembly {
  import sbtassembly.AssemblyUtils._
  import sbtassembly.Plugin._
  import AssemblyKeys._

  val hadoopVersion = taskKey[String]("The version of hadoop that spark is compiled against.")

  lazy val settings = assemblySettings ++ Seq(
    test in assembly := {},
    hadoopVersion := {
      sys.props.get("hadoop.version")
        .getOrElse(SbtPomKeys.effectivePom.value.getProperties.get("hadoop.version").asInstanceOf[String])
    },
    jarName in assembly := {
      if (moduleName.value.contains("streaming-flume-assembly")
        || moduleName.value.contains("streaming-kafka-0-8-assembly")
        || moduleName.value.contains("streaming-kafka-0-10-assembly")
        || moduleName.value.contains("streaming-kinesis-asl-assembly")) {
        // This must match the same name used in maven (see external/kafka-0-8-assembly/pom.xml)
        s"${moduleName.value}-${version.value}.jar"
      } else {
        s"${moduleName.value}-${version.value}-hadoop${hadoopVersion.value}.jar"
      }
    },
    jarName in (Test, assembly) := s"${moduleName.value}-test-${version.value}.jar",
    mergeStrategy in assembly := {
      case m if m.toLowerCase.endsWith("manifest.mf")          => MergeStrategy.discard
      case m if m.toLowerCase.matches("meta-inf.*\\.sf$")      => MergeStrategy.discard
      case "log4j.properties"                                  => MergeStrategy.discard
      case m if m.toLowerCase.startsWith("meta-inf/services/") => MergeStrategy.filterDistinctLines
      case "reference.conf"                                    => MergeStrategy.concat
      case _                                                   => MergeStrategy.first
    }
  )
}

object PySparkAssembly {
  import sbtassembly.Plugin._
  import AssemblyKeys._
  import java.util.zip.{ZipOutputStream, ZipEntry}

  lazy val settings = Seq(
    // Use a resource generator to copy all .py files from python/pyspark into a managed directory
    // to be included in the assembly. We can't just add "python/" to the assembly's resource dir
    // list since that will copy unneeded / unwanted files.
    resourceGenerators in Compile += Def.macroValueI(resourceManaged in Compile map { outDir: File =>
      val src = new File(BuildCommons.sparkHome, "python/pyspark")
      val zipFile = new File(BuildCommons.sparkHome , "python/lib/pyspark.zip")
      zipFile.delete()
      zipRecursive(src, zipFile)
      Seq.empty[File]
    }).value
  )

  private def zipRecursive(source: File, destZipFile: File) = {
    val destOutput = new ZipOutputStream(new FileOutputStream(destZipFile))
    addFilesToZipStream("", source, destOutput)
    destOutput.flush()
    destOutput.close()
  }

  private def addFilesToZipStream(parent: String, source: File, output: ZipOutputStream): Unit = {
    if (source.isDirectory()) {
      output.putNextEntry(new ZipEntry(parent + source.getName()))
      for (file <- source.listFiles()) {
        addFilesToZipStream(parent + source.getName() + File.separator, file, output)
      }
    } else {
      val in = new FileInputStream(source)
      output.putNextEntry(new ZipEntry(parent + source.getName()))
      val buf = new Array[Byte](8192)
      var n = 0
      while (n != -1) {
        n = in.read(buf)
        if (n != -1) {
          output.write(buf, 0, n)
        }
      }
      output.closeEntry()
      in.close()
    }
  }

}

object Unidoc {

  import BuildCommons._
  import sbtunidoc.Plugin._
  import UnidocKeys._

  private def ignoreUndocumentedPackages(packages: Seq[Seq[File]]): Seq[Seq[File]] = {
    packages
      .map(_.filterNot(_.getName.contains("$")))
      .map(_.filterNot(_.getCanonicalPath.contains("org/apache/spark/deploy")))
      .map(_.filterNot(_.getCanonicalPath.contains("org/apache/spark/examples")))
      .map(_.filterNot(_.getCanonicalPath.contains("org/apache/spark/memory")))
      .map(_.filterNot(_.getCanonicalPath.contains("org/apache/spark/network")))
      .map(_.filterNot(_.getCanonicalPath.contains("org/apache/spark/shuffle")))
      .map(_.filterNot(_.getCanonicalPath.contains("org/apache/spark/executor")))
      .map(_.filterNot(_.getCanonicalPath.contains("org/apache/spark/unsafe")))
      .map(_.filterNot(_.getCanonicalPath.contains("python")))
      .map(_.filterNot(_.getCanonicalPath.contains("org/apache/spark/util/collection")))
      .map(_.filterNot(_.getCanonicalPath.contains("org/apache/spark/util/kvstore")))
      .map(_.filterNot(_.getCanonicalPath.contains("org/apache/spark/sql/catalyst")))
      .map(_.filterNot(_.getCanonicalPath.contains("org/apache/spark/sql/execution")))
      .map(_.filterNot(_.getCanonicalPath.contains("org/apache/spark/sql/internal")))
      .map(_.filterNot(_.getCanonicalPath.contains("org/apache/spark/sql/hive/test")))
  }

  private def ignoreClasspaths(classpaths: Seq[Classpath]): Seq[Classpath] = {
    classpaths
      .map(_.filterNot(_.data.getCanonicalPath.matches(""".*kafka-clients-0\.10.*""")))
      .map(_.filterNot(_.data.getCanonicalPath.matches(""".*kafka_2\..*-0\.10.*""")))
  }

  val unidocSourceBase = settingKey[String]("Base URL of source links in Scaladoc.")

  lazy val settings = scalaJavaUnidocSettings ++ Seq (
    publish := {},

    unidocProjectFilter in(ScalaUnidoc, unidoc) :=
      inAnyProject -- inProjects(OldDeps.project, repl, examples, tools, streamingFlumeSink, kubernetes,
        yarn, tags, streamingKafka010, sqlKafka010, avro),
    unidocProjectFilter in(JavaUnidoc, unidoc) :=
      inAnyProject -- inProjects(OldDeps.project, repl, examples, tools, streamingFlumeSink, kubernetes,
        yarn, tags, streamingKafka010, sqlKafka010, avro),

    unidocAllClasspaths in (ScalaUnidoc, unidoc) := {
      ignoreClasspaths((unidocAllClasspaths in (ScalaUnidoc, unidoc)).value)
    },

    unidocAllClasspaths in (JavaUnidoc, unidoc) := {
      ignoreClasspaths((unidocAllClasspaths in (JavaUnidoc, unidoc)).value)
    },

    // Skip actual catalyst, but include the subproject.
    // Catalyst is not public API and contains quasiquotes which break scaladoc.
    unidocAllSources in (ScalaUnidoc, unidoc) := {
      ignoreUndocumentedPackages((unidocAllSources in (ScalaUnidoc, unidoc)).value)
    },

    // Skip class names containing $ and some internal packages in Javadocs
    unidocAllSources in (JavaUnidoc, unidoc) := {
      ignoreUndocumentedPackages((unidocAllSources in (JavaUnidoc, unidoc)).value)
        .map(_.filterNot(_.getCanonicalPath.contains("org/apache/hadoop")))
    },

    javacOptions in (JavaUnidoc, unidoc) := Seq(
      "-windowtitle", "Spark " + version.value.replaceAll("-SNAPSHOT", "") + " JavaDoc",
      "-public",
      "-noqualifier", "java.lang",
      "-tag", """example:a:Example\:""",
      "-tag", """note:a:Note\:""",
      "-tag", "group:X",
      "-tag", "tparam:X",
      "-tag", "constructor:X",
      "-tag", "todo:X",
      "-tag", "groupname:X"
    ),

    // Use GitHub repository for Scaladoc source links
    unidocSourceBase := s"https://github.com/apache/spark/tree/v${version.value}",

    scalacOptions in (ScalaUnidoc, unidoc) ++= Seq(
      "-groups", // Group similar methods together based on the @group annotation.
      "-skip-packages", "org.apache.hadoop",
      "-sourcepath", (baseDirectory in ThisBuild).value.getAbsolutePath
    ) ++ (
      // Add links to sources when generating Scaladoc for a non-snapshot release
      if (!isSnapshot.value) {
        Opts.doc.sourceUrl(unidocSourceBase.value + "€{FILE_PATH}.scala")
      } else {
        Seq()
      }
    )
  )
}

object Checkstyle {
  lazy val settings = Seq(
    checkstyleSeverityLevel := Some(CheckstyleSeverityLevel.Error),
    javaSource in (Compile, checkstyle) := baseDirectory.value / "src/main/java",
    javaSource in (Test, checkstyle) := baseDirectory.value / "src/test/java",
    checkstyleConfigLocation := CheckstyleConfigLocation.File("dev/checkstyle.xml"),
    checkstyleOutputFile := baseDirectory.value / "target/checkstyle-output.xml",
    checkstyleOutputFile in Test := baseDirectory.value / "target/checkstyle-output.xml"
  )
}

object CopyDependencies {

  val copyDeps = TaskKey[Unit]("copyDeps", "Copies needed dependencies to the build directory.")
  val destPath = (crossTarget in Compile) { _ / "jars"}

  lazy val settings = Seq(
    copyDeps := {
      val dest = destPath.value
      if (!dest.isDirectory() && !dest.mkdirs()) {
        throw new IOException("Failed to create jars directory.")
      }

      (dependencyClasspath in Compile).value.map(_.data)
        .filter { jar => jar.isFile() }
        .foreach { jar =>
          val destJar = new File(dest, jar.getName())
          if (destJar.isFile()) {
            destJar.delete()
          }
          Files.copy(jar.toPath(), destJar.toPath())
        }
    },
    crossTarget in (Compile, packageBin) := destPath.value,
    packageBin in Compile := (packageBin in Compile).dependsOn(copyDeps).value
  )

}

object TestSettings {
  import BuildCommons._

  private val scalaBinaryVersion =
    if (System.getProperty("scala-2.12") == "true") {
      "2.12"
    } else {
      "2.11"
    }
  lazy val settings = Seq (
    // Fork new JVMs for tests and set Java options for those
    fork := true,
    // Setting SPARK_DIST_CLASSPATH is a simple way to make sure any child processes
    // launched by the tests have access to the correct test-time classpath.
    envVars in Test ++= Map(
      "SPARK_DIST_CLASSPATH" ->
        (fullClasspath in Test).value.files.map(_.getAbsolutePath)
          .mkString(File.pathSeparator).stripSuffix(File.pathSeparator),
      "SPARK_PREPEND_CLASSES" -> "1",
      "SPARK_SCALA_VERSION" -> scalaBinaryVersion,
      "SPARK_TESTING" -> "1",
      "JAVA_HOME" -> sys.env.get("JAVA_HOME").getOrElse(sys.props("java.home"))),
    javaOptions in Test += s"-Djava.io.tmpdir=$testTempDir",
    javaOptions in Test += "-Dspark.test.home=" + sparkHome,
    javaOptions in Test += "-Dspark.testing=1",
    javaOptions in Test += "-Dspark.port.maxRetries=100",
    javaOptions in Test += "-Dspark.master.rest.enabled=false",
    javaOptions in Test += "-Dspark.memory.debugFill=true",
    javaOptions in Test += "-Dspark.ui.enabled=false",
    javaOptions in Test += "-Dspark.ui.showConsoleProgress=false",
    javaOptions in Test += "-Dspark.unsafe.exceptionOnMemoryLeak=true",
    javaOptions in Test += "-Dsun.io.serialization.extendedDebugInfo=false",
    javaOptions in Test += "-Dderby.system.durability=test",
    javaOptions in Test ++= System.getProperties.asScala.filter(_._1.startsWith("spark"))
      .map { case (k,v) => s"-D$k=$v" }.toSeq,
    javaOptions in Test += "-ea",
    javaOptions in Test ++= "-Xmx3g -Xss4m"
      .split(" ").toSeq,
    javaOptions += "-Xmx3g",
    // Exclude tags defined in a system property
    testOptions in Test += Tests.Argument(TestFrameworks.ScalaTest,
      sys.props.get("test.exclude.tags").map { tags =>
        tags.split(",").flatMap { tag => Seq("-l", tag) }.toSeq
      }.getOrElse(Nil): _*),
    testOptions in Test += Tests.Argument(TestFrameworks.JUnit,
      sys.props.get("test.exclude.tags").map { tags =>
        Seq("--exclude-categories=" + tags)
      }.getOrElse(Nil): _*),
    // Show full stack trace and duration in test cases.
    testOptions in Test += Tests.Argument("-oDF"),
    testOptions in Test += Tests.Argument(TestFrameworks.JUnit, "-v", "-a"),
    // Enable Junit testing.
    libraryDependencies += "com.novocode" % "junit-interface" % "0.11" % "test",
    // `parallelExecutionInTest` controls whether test suites belonging to the same SBT project
    // can run in parallel with one another. It does NOT control whether tests execute in parallel
    // within the same JVM (which is controlled by `testForkedParallel`) or whether test cases
    // within the same suite can run in parallel (which is a ScalaTest runner option which is passed
    // to the underlying runner but is not a SBT-level configuration). This needs to be `true` in
    // order for the extra parallelism enabled by `SparkParallelTestGrouping` to take effect.
    // The `SERIAL_SBT_TESTS` check is here so the extra parallelism can be feature-flagged.
    parallelExecution in Test := { if (sys.env.contains("SERIAL_SBT_TESTS")) false else true },
    // Make sure the test temp directory exists.
    resourceGenerators in Test += Def.macroValueI(resourceManaged in Test map { outDir: File =>
      var dir = new File(testTempDir)
      if (!dir.isDirectory()) {
        // Because File.mkdirs() can fail if multiple callers are trying to create the same
        // parent directory, this code tries to create parents one at a time, and avoids
        // failures when the directories have been created by somebody else.
        val stack = new Stack[File]()
        while (!dir.isDirectory()) {
          stack.push(dir)
          dir = dir.getParentFile()
        }

        while (stack.nonEmpty) {
          val d = stack.pop()
          require(d.mkdir() || d.isDirectory(), s"Failed to create directory $d")
        }
      }
      Seq.empty[File]
    }).value,
    concurrentRestrictions in Global := {
      // The number of concurrent test groups is empirically chosen based on experience
      // with Jenkins flakiness.
      if (sys.env.contains("SERIAL_SBT_TESTS")) (concurrentRestrictions in Global).value
      else Seq(Tags.limit(Tags.ForkedTestGroup, 4))
    }
  )

}
