package common

import (
	"bytes"
	"fmt"
	rayiov1alpha1 "ray-operator/api/v1alpha1"
	"strings"

	"k8s.io/apimachinery/pkg/types"

	logf "sigs.k8s.io/controller-runtime/pkg/runtime/log"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var log = logf.Log.WithName("RayCluster-Controller")

const (
	defaultServiceAccountName = "default"
)

type PodConfig struct {
	RayCluster  *rayiov1alpha1.RayCluster
	PodTypeName string
	PodName     string
	podTemplate *v1.PodTemplateSpec
}

// DefaultPodConfig to be removed
func DefaultPodConfig(instance *rayiov1alpha1.RayCluster, podTypeName string, podName string) *PodConfig {
	return &PodConfig{
		RayCluster:  instance,
		PodTypeName: podTypeName,
		PodName:     podName,
	}
}

// DefaultHeadPodConfig sets the config values
func DefaultHeadPodConfig(instance *rayiov1alpha1.RayCluster, podTypeName string, podName string) *PodConfig {
	podTemplate := &instance.Spec.HeadGroupSpec.Template
	podTemplate.ObjectMeta = instance.Spec.HeadGroupSpec.Template.ObjectMeta
	podTemplate.Spec = instance.Spec.HeadGroupSpec.Template.Spec
	pConfig := &PodConfig{
		RayCluster:  instance,
		PodTypeName: podTypeName,
		PodName:     podName,
		podTemplate: podTemplate,
	}
	if pConfig.podTemplate.Labels == nil {
		pConfig.podTemplate.Labels = make(map[string]string)
	}
	pConfig.podTemplate.Labels = labelPod("head", instance.Name, "headGroup", instance.Spec.HeadGroupSpec.Template.ObjectMeta.Labels)

	if pConfig.podTemplate.ObjectMeta.Namespace == "" {
		pConfig.podTemplate.ObjectMeta.Namespace = instance.Namespace
		log.Info("Setting pod namespaces", "namespace", instance.Namespace)
	}

	pConfig.podTemplate.Name = podName

	return pConfig
}

// todo verify the values here

// DefaultWorkerPodConfig sets the config values
func DefaultWorkerPodConfig(instance *rayiov1alpha1.RayCluster, workerSpec *rayiov1alpha1.WorkerGroupSpec, podTypeName string, podName string) *PodConfig {
	podTemplate := &workerSpec.Template
	podTemplate.ObjectMeta = workerSpec.Template.ObjectMeta
	podTemplate.Spec = workerSpec.Template.Spec
	pConfig := &PodConfig{
		RayCluster:  instance,
		PodTypeName: podTypeName,
		PodName:     podName,
		podTemplate: podTemplate,
	}
	if pConfig.podTemplate.Labels == nil {
		pConfig.podTemplate.Labels = make(map[string]string)
	}
	pConfig.podTemplate.Labels = labelPod("worker", instance.Name, workerSpec.GroupName, workerSpec.Template.ObjectMeta.Labels)

	if pConfig.podTemplate.ObjectMeta.Namespace == "" {
		pConfig.podTemplate.ObjectMeta.Namespace = instance.Namespace
		log.Info("Setting pod namespaces", "namespace", instance.Namespace)
	}

	pConfig.podTemplate.Name = podName

	return pConfig
}

// BuildPod a pod config
func BuildPod(conf *PodConfig, rayNodeType rayiov1alpha1.RayNodeType, rayStartParams map[string]string) *v1.Pod {

	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: conf.podTemplate.ObjectMeta,
		Spec:       conf.podTemplate.Spec,
	}
	index := getRayContainerIndex(pod)
	cont := concatinateContainerCommand(rayNodeType, rayStartParams)
	// using a single line command in the container
	// i want args to be ulimit + ray start + oldCOmmnand
	// i want command to be bash -c

	//saving temporarly the old command and args
	var cmd, args string
	if len(pod.Spec.Containers[index].Command) > 0 {
		cmd = convertCmdToString(pod.Spec.Containers[index].Command)
	}
	if len(pod.Spec.Containers[index].Args) > 0 {
		cmd += convertCmdToString(pod.Spec.Containers[index].Args)
	}
	// replacing the old command
	pod.Spec.Containers[index].Command = []string{"/bin/bash", "-c", "--"}
	if cmd != "" {
		args = fmt.Sprintf("%s;%s", cont, cmd)
	} else {
		args = fmt.Sprintf("%s", cont)
	}

	pod.Spec.Containers[index].Args = []string{args}
	/*
		if pod.Spec.Containers[index].Args == nil || len(pod.Spec.Containers[index].Args) == 0 {
			pod.Spec.Containers[index].Command = []string{""}
			pod.Spec.Containers[index].Command[0] = fmt.Sprintf("%s", cont)
		} else {
			Args := fmt.Sprintf("%s;%s", cont, convertCmdToString(pod.Spec.Containers[index].Command))
			pod.Spec.Containers[index].Command[0] = fmt.Sprintf("%s;%s", cont, convertCmdToString(pod.Spec.Containers[index].Command))

		}
	*/
	return pod
}

func convertCmdToString(cmdArr []string) (cmd string) {
	cmdAggr := new(bytes.Buffer)
	for _, v := range cmdArr {
		fmt.Fprintf(cmdAggr, " %s ", v)
	}
	return cmdAggr.String()

}

func getRayContainerIndex(pod *v1.Pod) (index int) {
	// theoretically, a ray pod can have multiple containers.
	// we identify the ray container based on env var: RAY=true
	// if the env var is missing, we choose containers[0].
	for i, container := range pod.Spec.Containers {
		for _, env := range container.Env {
			if env.Name == strings.ToLower("ray") && env.Value == strings.ToLower("true") {
				return i
			}
		}
	}
	//not found, use first container
	return 0
}

// The function labelsForCluster returns the labels for selecting the resources
// belonging to the given RayCluster CR name.
func labelPod(rayNodeType string, rayClusterName string, groupName string, labels map[string]string) (ret map[string]string) {
	ret = map[string]string{
		rayClusterName: rayClusterName,
		rayNodeType:    rayNodeType,
		groupName:      groupName,
	}

	for k, v := range ret {
		if _, ok := labels[k]; !ok {
			labels[k] = v
		}
	}
	return labels
}

//TODO set container extra env vars, such as head service IP and port
func setContainerEnvVars(container *v1.Container, head bool, svcName types.NamespacedName, rayStartParams map[string]string) {
	// set IP to local host if head, or the the svc otherwise  RAY_IP
	// set the port RAY_PORT
	// set the password?
	if container.Env == nil || len(container.Env) == 0 {
		container.Env = []v1.EnvVar{}
	}
	if !envVarExists("RAY_IP", container.Env) {
		ip := v1.EnvVar{Name: "RAY_IP"}
		if head {
			// if head, use localhost
			ip.Value = "127.0.0.1"
		} else {
			// if worker, use the service name of the head
			ip.Value = fmt.Sprintf("%s.%s", svcName.Namespace, svcName.Name)
		}
		container.Env = append(container.Env, ip)
	}
	if !envVarExists("RAY_PORT", container.Env) {
		port := v1.EnvVar{Name: "RAY_PORT"}
		if value, ok := rayStartParams["port"]; !ok {
			// using default port
			port.Value = "6379"
		} else {
			// setting the RAY_PORT env var from the params
			port.Value = value
		}
		container.Env = append(container.Env, port)
	}
	if !envVarExists("REDIS_PASSWORD", container.Env) {
		// setting the REDIS_PASSWORD env var from the params
		port := v1.EnvVar{Name: "REDIS_PASSWORD"}
		if value, ok := rayStartParams["redis-password"]; ok {
			port.Value = value
		}
		container.Env = append(container.Env, port)
	}

}

func envVarExists(envName string, envVars []v1.EnvVar) bool {
	for _, env := range envVars {
		if env.Name == envName {
			return true
		}

	}
	return false

}

//TODO auto complete params
func setMissingRayStartParams() {
	// set the # of CPU
}

//TODO concatinateContainerCommand with ray start
func concatinateContainerCommand(nodeType rayiov1alpha1.RayNodeType, rayStartParams map[string]string) (fullCmd string) {
	switch nodeType {
	case rayiov1alpha1.HeadNode:
		return fmt.Sprintf("ulimit -n 65536; ray start --head %s", convertParamMap(rayStartParams))
	case rayiov1alpha1.WorkerNode:
		return fmt.Sprintf("ulimit -n 65536; ray start --block %s", convertParamMap(rayStartParams))
	default:
		log.Error(fmt.Errorf("missing node type"), "a node must be either head or worker")
	}
	return ""
}

func convertParamMap(rayStartParams map[string]string) (s string) {
	flags := new(bytes.Buffer)
	for k, v := range rayStartParams {
		fmt.Fprintf(flags, " --%s=%s ", k, v)
	}
	return flags.String()
}

/*
head_start_ray_commands:
    - ray stop
    - ulimit -n 65536; ray start --head --num-cpus=$MY_CPU_REQUEST --port=6379 --object-manager-port=8076 --autoscaling-config=~/ray_bootstrap_config.yaml --dashboard-host 0.0.0.0

# Command to start ray on worker nodes. You don't need to change this.
worker_start_ray_commands:
    - ray stop
    - ulimit -n 65536; ray start --num-cpus=$MY_CPU_REQUEST --address=$RAY_HEAD_IP:6379 --object-manager-port=8076

*/
