pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        // Use the alternate Maven Central hostname for ONNX Runtime Android.
        // Restrict it to this group; all other dependencies keep the normal
        // Google/Maven Central resolution order.
        maven {
            url = uri("https://repo1.maven.org/maven2")
            content {
                includeGroup("com.microsoft.onnxruntime")
            }
        }
        mavenCentral()
    }
}
rootProject.name = "TataAI"
include(":app", ":onnx-poc")
