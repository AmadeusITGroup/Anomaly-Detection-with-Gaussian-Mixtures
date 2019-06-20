name := "gmm-scoring"

organization := "com.amadeus"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark"         %% "spark-core"            % "2.1.0" % "provided",
  "org.apache.spark"         %% "spark-sql"             % "2.1.0" % "provided",
  "org.apache.spark"         %% "spark-mllib"           % "2.1.0" % "provided",
  "org.scalactic"            %% "scalactic"             % "3.0.1" % "test",
  "org.scalatest"            %% "scalatest"             % "3.0.1" % "test",
  "joda-time"                % "joda-time"              % "2.9.3",
  "org.joda"                 % "joda-convert"           % "1.8.1"
)

resolvers += "amadeus-mvn" at "https://repository.rnd.amadeus.net/mvn-built"
resolvers += "amadeus-sbt" at "https://repository.rnd.amadeus.net/sbt-built"
