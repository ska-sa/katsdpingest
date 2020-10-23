#!groovy

@Library('katsdpjenkins') _
katsdp.killOldJobs()

katsdp.setDependencies([
    'ska-sa/katsdpsigproc/master',
    'ska-sa/katsdpdockerbase/new-rdma-core',
    'ska-sa/katsdpservices/master',
    'ska-sa/katsdptelstate/master',
    'ska-sa/katsdpmodels/master',
    'ska-sa/katdal'])

catchError {
    katsdp.stagePrepare(python2: false, python3: true,
                        timeout: [time: 60, unit: 'MINUTES'],
                        katsdpdockerbase_ref: 'new-rdma-core')
    katsdp.stageNosetestsGpu(cuda: true, opencl: true)
    katsdp.stageFlake8()
    katsdp.stageMypy()
    katsdp.stageMakeDocker(venv: true)

    stage('katsdpingest/autotuning') {
        if (katsdp.notYetFailed()) {
            katsdp.simpleNode(label: 'cuda-GeForce_GTX_TITAN_X') {
                deleteDir()
                katsdp.unpackGit()
                katsdp.unpackVenv()
                katsdp.unpackKatsdpdockerbase()
                katsdp.virtualenv('venv') {
                    dir('git') {
                        lock("katsdpingest-autotune-${env.BRANCH_NAME}") {
                            sh './jenkins-autotune.sh geforce_gtx_titan_x'
                        }
                    }
                }
            }
        }
    }
}
katsdp.mail('sdpdev+katsdpingest@ska.ac.za')
