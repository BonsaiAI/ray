package common

import (
	rayiov1alpha1 "ray-operator/api/v1alpha1"
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type ServiceConfig struct {
	RayCluster rayiov1alpha1.RayCluster
	PodName    string
}

func DefaultServiceConfig(instance rayiov1alpha1.RayCluster, podName string) *ServiceConfig {
	return &ServiceConfig{
		RayCluster: instance,
		PodName:    podName,
	}
}

// Build the service for a pod. Currently, there is only one service that allows
// the worker nodes to connect to the head node.
func ServiceForPod(instance rayiov1alpha1.RayCluster, conf *ServiceConfig) *corev1.Service {
	//TODO add prefix the Ray cluster name
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      instance.Spec.HeadService.Name,
			Namespace: instance.Spec.HeadService.Namespace,
		},
		Spec: corev1.ServiceSpec{
			Ports: []corev1.ServicePort{{Name: "redis", Port: int32(defaultRedisPort)}},
			// Use a headless service, meaning that the DNS record for the service will
			// point directly to the head node pod's IP address.
			ClusterIP: corev1.ClusterIPNone,
			// This selector must match the label of the head node.
			Selector: map[string]string{
				rayclusterComponent: conf.PodName,
			},
		},
	}

	return svc
}

//TODO change this logic
func checkSvcName(instanceName, svcName string) (name string) {
	if strings.Contains(svcName, instanceName) {
		return svcName
	}
	return svcName
}
