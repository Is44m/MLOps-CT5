pipeline {
    agent {
        label 'ubuntu' // Specify GitHub runner as the agent
    }

    environment {
        DOCKER_USERNAME = credentials('docker-username') // Jenkins credentials ID for Docker username
        DOCKER_PASSWORD = credentials('docker-password') // Jenkins credentials ID for Docker password
    }

    stages {
        stage('Checkout Repository') {
            steps {
                // Checkout the repository
                checkout scm
            }
        }

        stage('Set up Docker Buildx') {
            steps {
                sh 'docker run --rm --privileged docker/binfmt:a7996909642ee92942dcd6cff44b9b95f08dad64'
            }
        }

        stage('Log in to Docker Hub') {
            steps {
                sh 'echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin'
            }
        }

        stage('Build and Push Docker Image') {
            steps {
                sh '''
                    docker buildx create --use
                    docker buildx build --platform linux/amd64,linux/arm64 --push -t $DOCKER_USERNAME/my-flask-app:latest .
                '''
            }
        }
    }
}
