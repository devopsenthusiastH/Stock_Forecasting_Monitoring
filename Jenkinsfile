pipeline {
    agent any
    
    environment{
        SCANNER_HOME= tool 'sonar-scanner'
    }

    stages {
        stage('Git Checkout') {
            steps {
                git branch: 'main', credentialsId: 'git-credentials', url: 'https://github.com/devopsenthusiastH/Stock_Forecasting_Monitoring.git'
            }
        }
        stage('Trivy File System Scan') {
            steps {
                sh "trivy fs --format table -o trivy-fs-report.html ."
            }
        }
        stage('SonarQube Analysis') {
            steps {
                withSonarQubeEnv('sonar') {
                    sh ''' $SCANNER_HOME/bin/sonar-scanner -Dsonar.projectName=sfm -Dsonar.projectKey=sfm '''
                }
            }
        }
        stage('Qualtiy gates') {
            steps {
                script {
                    waitForQualityGate abortPipeline: false, credentialsId: 'token'
                }
            }
        }
        stage('Docker Build') {
            steps {
                script {
                    withDockerRegistry(credentialsId: 'docker-cred', toolName: 'docker') {
                        sh "docker build -t aakashhandibar/stock_forecasting:v2.1 ."
                    }
                }
            }
        }
        stage('Trivy Docker Image Scan') {
            steps {
                sh "trivy image --format table -o trivy-image-report.html aakashhandibar/stock_forecasting:v2.1"
            }
        }
        stage('Docker Push') {
            steps {
                script {
                    withDockerRegistry(credentialsId: 'docker-cred', toolName: 'docker') {
                        sh "docker push aakashhandibar/stock_forecasting:v2.1"
                        sh "docker images"
                        sh "docker rmi aakashhandibar/stock_forecasting:v2.1"
                    }
                }
            }
        }
        stage('Deploy to kubernetes') {
            steps {
                withKubeConfig(caCertificate: '', clusterName: 'eks-sfm', contextName: '', credentialsId: 'kube-secret', namespace: 'sfm', restrictKubeConfigAccess: false, serverUrl: 'https://9D6B2BA7A8966ACFE312ECE75277B58B.gr7.ap-south-1.eks.amazonaws.com') {
                    dir('/var/lib/jenkins/workspace/sfm/k8smanifests') {
                        sh "kubectl apply -f deployment.yaml"
                        sh "kubectl apply -f service.yaml"
                        sh "kubectl get pods"
                        sh "kubectl get svc"
                    }
                }
            }
        }
    }
}
