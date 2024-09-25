pipeline {
    agent any

    environment {
        DOCKER_CREDENTIALS_ID = 'docker-credentials' // ID of your Jenkins credentials
        IMAGE_NAME = 'jenkins-d-image' 
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
                sh 'docker buildx create --use || true'  // Ensure Docker Buildx is available
            }
        }

        stage('Log in to Docker Hub') {
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: DOCKER_CREDENTIALS_ID, usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                        sh """
                        echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin
                        """
                    }
                }
            }
        }

        stage('Build and push Docker image') {
            steps {
                script {
                    sh """
                    docker build -t $DOCKER_USERNAME/${IMAGE_NAME}:${DOCKER_TAG} .
                    docker push $DOCKER_USERNAME/${IMAGE_NAME}:${DOCKER_TAG}
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
