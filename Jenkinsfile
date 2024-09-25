pipeline {
    agent any

    environment {
        DOCKER_CREDENTIALS_ID = 'docker-credentials'
        IMAGE_NAME = 'my-docker-image'
        DOCKER_TAG = 'latest'
    }

    stages {
        stage('Checkout repository') {
            steps {
                // Checkout the Git repository
                checkout scm
            }
        }

        stage('Set up Docker Buildx') {
            steps {
                // Replace with relevant Windows commands
                bat '''
                    docker buildx create --use || exit /B 0
                '''
            }
        }

        stage('Log in to Docker Hub') {
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: DOCKER_CREDENTIALS_ID, usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                        bat """
                            echo %DOCKER_PASSWORD% | docker login -u %DOCKER_USERNAME% --password-stdin
                        """
                    }
                }
            }
        }

        stage('Build and push Docker image') {
            steps {
                script {
                    bat """
                        docker build -t %DOCKER_USERNAME%/${IMAGE_NAME}:${DOCKER_TAG} .
                        docker push %DOCKER_USERNAME%/${IMAGE_NAME}:${DOCKER_TAG}
                    """
                }
            }
        }
    }

    post {
        always {
            echo 'Pipeline execution completed.'
        }
        failure {
            echo 'Pipeline execution failed.'
        }
    }
}
