#!groovy

@Library('katsdpjenkins@install_pinned_ng') _
katsdp.killOldJobs()

katsdp.setDependencies([
    'ska-sa/katsdpsigproc/install_pinned_ng',
    'ska-sa/katsdpdockerbase/master',
    'ska-sa/katsdpservices/master',
    'ska-sa/katsdptelstate/master',
    'ska-sa/katsdpmodels/master',
    'ska-sa/katdal/master',
    'ska-sa/katpoint/master'])

catchError {
    katsdp.standardBuild(
        cuda: true,
        opencl: true,
        prepare_timeout: [time: 60, unit: 'MINUTES'],
        docker_venv: true,
        katsdpdockerbase_ref: 'refs/heads/install_pinned_ng'
    )
    stage('katsdpingest/autotuning') {
        if (katsdp.notYetFailed()) {
            katsdp.simpleNode(label: 'cuda-A30', timeout: [time: 60, unit: 'MINUTES']) {
                deleteDir()
                katsdp.unpackGit()
                katsdp.unpackVenv()
                katsdp.unpackKatsdpdockerbase()
                withCredentials([usernamePassword(
                    credentialsId: 'harbor-dpp',
                    usernameVariable: 'HARBOR_USER',
                    passwordVariable: 'HARBOR_PASS')]) {
                    sh 'docker login -u "$HARBOR_USER" -p "$HARBOR_PASS" "harbor.sdp.kat.ac.za"'
                }
                katsdp.virtualenv('venv') {
                    dir('git') {
                        lock("katsdpingest-autotune-${env.BRANCH_NAME}") {
                            sh './jenkins-autotune.sh a30'
                        }
                    }
                }
            }
        }
    }
}
katsdp.mail('sdpdev+katsdpingest@ska.ac.za')
