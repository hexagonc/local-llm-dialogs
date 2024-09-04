import subprocess
import os
import shutil


def create_kotlin_compose_project():
    project_name = "graph-demo"
    project_path = os.path.join(os.getcwd(), project_name)

    # Create the project directory
    os.makedirs(project_path, exist_ok=True)

    # Create basic project structure
    os.makedirs(os.path.join(project_path, "app/src/main/java/com/example/graphdemo"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "app/src/main/res/layout"), exist_ok=True)

    # Create settings.gradle
    with open(os.path.join(project_path, "settings.gradle"), "w") as f:
        f.write(f"include ':app'\n")

    # Create build.gradle files
    with open(os.path.join(project_path, "build.gradle"), "w") as f:
        f.write("""
buildscript {
    ext.kotlin_version = '1.5.31'
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath 'com.android.tools.build:gradle:7.0.2'
        classpath "org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlin_version"
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}
        """)

    # Use the provided build.gradle content for the app module
    build_gradle_content = """
plugins {
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
}

android {
    namespace 'com.example.graphdemo'
    compileSdk 33

    defaultConfig {
        applicationId "com.example.graphdemo"
        minSdk 24
        targetSdk 33
        versionCode 1
        versionName "1.0"

        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary true
        }
    }

    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = '1.8'
    }
    buildFeatures {
        compose true
    }
    composeOptions {
        kotlinCompilerExtensionVersion '1.3.2'
    }
    packagingOptions {
        resources {
            excludes += '/META-INF/{AL2.0,LGPL2.1}'
        }
    }
}

dependencies {

    implementation 'androidx.core:core-ktx:1.8.0'
    implementation 'androidx.lifecycle:lifecycle-runtime-ktx:2.3.1'
    implementation 'androidx.activity:activity-compose:1.5.1'
    implementation platform('androidx.compose:compose-bom:2022.10.00')
    implementation 'androidx.compose.ui:ui'
    implementation 'androidx.compose.ui:ui-graphics'
    implementation 'androidx.compose.ui:ui-tooling-preview'
    implementation 'androidx.compose.material3:material3'
    testImplementation 'junit:junit:4.13.2'
    androidTestImplementation 'androidx.test.ext:junit:1.1.5'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.5.1'
    androidTestImplementation platform('androidx.compose:compose-bom:2022.10.00')
    androidTestImplementation 'androidx.compose.ui:ui-test-junit4'
    debugImplementation 'androidx.compose.ui:ui-tooling'
    debugImplementation 'androidx.compose.ui:ui-test-manifest'
}
    """
    with open(os.path.join(project_path, "app/build.gradle"), "w") as f:
        f.write(build_gradle_content)

    # Create a basic MainActivity.kt file
    main_activity_content = """
package com.example.graphdemo

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Surface
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.tooling.preview.Preview
import com.example.graphdemo.ui.theme.GraphDemoTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            GraphDemoTheme {
                Surface(color = MaterialTheme.colors.background) {
                    Greeting("Android")
                }
            }
        }
    }
}

@Composable
fun Greeting(name: String) {
    Text(text = "Hello $name!")
}

@Preview(showBackground = true)
@Composable
fun DefaultPreview() {
    GraphDemoTheme {
        Greeting("Android")
    }
}
"""
    with open(os.path.join(project_path, "app/src/main/java/com/example/graphdemo/MainActivity.kt"), "w") as f:
        f.write(main_activity_content)

    print(f"Kotlin Compose Android application project created in {project_path}")


if __name__ == "__main__":
    create_kotlin_compose_project()