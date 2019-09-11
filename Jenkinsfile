#!groovy

@Library('katsdpjenkins') _
katsdp.killOldJobs()

katsdp.setDependencies([
    'ska-sa/katsdpsigproc/master',
    'ska-sa/katsdpdockerbase/master',
    'ska-sa/katsdpservices/master',
    'ska-sa/katsdptelstate/master',
    'ska-sa/katdal'])

if (!katsdp.isAborted()) {
    catchError {
        katsdp.stagePrepare(subdir: 'katsdpingest', python2: false, python3: true,
                            timeout: [time: 60, unit: 'MINUTES'])
        katsdp.stageNosetestsGpu(subdir: 'katsdpingest', cuda: true, opencl: true)
        katsdp.stageFlake8(subdir: 'katsdpingest')
        katsdp.stageMypy(subdir: 'katsdpingest')
        katsdp.stageMakeDocker(subdir: 'katsdpingest', venv: true)

        stage('katsdpingest/autotuning') {
            if (katsdp.notYetFailed()) {
                katsdp.simpleNode(label: 'cuda-GeForce_GTX_TITAN_X') {
                    deleteDir()
                    katsdp.unpackGit()
                    katsdp.unpackVenv()
                    katsdp.unpackKatsdpdockerbase()
                    katsdp.virtualenv('venv3') {
                        dir('git/katsdpingest') {
                            lock("katsdpingest-autotune-${env.BRANCH_NAME}") {
                                sh './jenkins-autotune.sh titanx'
                            }
                        }
                    }
                }
            }
        }
    }
}
katsdp.mail('bmerry@ska.ac.za')
