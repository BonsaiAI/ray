package controllers

import (
	"context"
	"fmt"
	rayiov1alpha1 "ray-operator/api/v1alpha1"
	"ray-operator/controllers/common"
	_ "ray-operator/controllers/common"
	"ray-operator/controllers/utils"
	"strings"
	"time"

	mapset "github.com/deckarep/golang-set"
	"github.com/go-logr/logr"
	_ "k8s.io/api/apps/v1beta1"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	logf "sigs.k8s.io/controller-runtime/pkg/runtime/log"
	"sigs.k8s.io/controller-runtime/pkg/source"
)

// K8sClient client used query K8s outside the RayClusterReconciler
var K8sClient client.Client
var log = logf.Log.WithName("RayCluster-Controller")

// newReconciler returns a new reconcile.Reconciler
func newReconciler(mgr manager.Manager) reconcile.Reconciler {
	return &RayClusterReconciler{Client: mgr.GetClient(), Scheme: mgr.GetScheme()}
}

var _ reconcile.Reconciler = &RayClusterReconciler{}

// RayClusterReconciler reconciles a RayCluster object
type RayClusterReconciler struct {
	client.Client
	Log    logr.Logger
	Scheme *runtime.Scheme
}

// Reconcile reads that state of the cluster for a RayCluster object and makes changes based on it
// and what is in the RayCluster.Spec
// Automatically generate RBAC rules to allow the Controller to read and write workloads
// +kubebuilder:rbac:groups=ray.io,resources=rayclusters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=ray.io,resources=rayclusters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=core,resources=events,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=pods/status,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch;create;update;patch;delete

// Reconcile used to bridge the desired state with the current state
func (r *RayClusterReconciler) Reconcile(request reconcile.Request) (reconcile.Result, error) {
	_ = r.Log.WithValues("raycluster", request.NamespacedName)
	log.Info("Reconciling RayCluster", "cluster name", request.Name)

	// Fetch the RayCluster instance
	instance := &rayiov1alpha1.RayCluster{}
	err := r.Get(context.TODO(), request.NamespacedName, instance)

	if err != nil {
		if errors.IsNotFound(err) {
			// Object not found, return.  Created objects are automatically garbage collected.
			// For additional cleanup logic use finalizers.
			return reconcile.Result{}, nil
		}
		log.Error(err, "Read request instance error!")
		// Error reading the object - requeue the request.
		if !apierrs.IsNotFound(err) {
			return reconcile.Result{}, err
		}
		return reconcile.Result{}, nil
	}

	if instance.Spec.HeadService.Namespace == "" {
		if instance.Namespace != "" {
			// the Custom resource namespace is assumed to be the same for all the pods and the head service.
			instance.Spec.HeadService.Namespace = instance.Namespace
		} else {
			instance.Spec.HeadService.Namespace = "default"
		}
	}
	// adding prefix of cluster-name
	//TODO check the name format before changing it
	instance.Spec.HeadService.Name = fmt.Sprintf("%s-%s", instance.Name, instance.Spec.HeadService.Name)
	svcName := types.NamespacedName{
		Name:      instance.Spec.HeadService.Name,
		Namespace: instance.Spec.HeadService.Namespace,
	}

	// Build pods for instance
	expectedPods := r.buildHeadPods(instance, svcName)
	expectedPods = append(expectedPods, r.buildWorkerPods(instance, svcName)...)

	expectedPodNameList := mapset.NewSet()
	expectedPodMap := make(map[string]corev1.Pod)
	needServicePodMap := mapset.NewSet()
	for _, pod := range expectedPods {
		expectedPodNameList.Add(pod.Name)
		expectedPodMap[pod.Name] = pod
		if strings.EqualFold(pod.Labels["rayNodeType"], string(rayiov1alpha1.HeadNode)) {
			needServicePodMap.Add(pod.Name)
		}
	}

	log.Info("Build pods according to the ray cluster instance", "size", len(expectedPods), "podNames", expectedPodNameList)

	runtimePods := corev1.PodList{}
	if err = r.List(context.TODO(), &runtimePods, client.InNamespace(instance.Namespace), client.MatchingLabels{common.RayClusterOwnerKey: request.Name}); err != nil {
		return reconcile.Result{}, err
	}

	runtimePodNameList := mapset.NewSet()
	runtimePodMap := make(map[string]corev1.Pod)
	for _, runtimePod := range runtimePods.Items {
		runtimePodNameList.Add(runtimePod.Name)
		runtimePodMap[runtimePod.Name] = runtimePod
	}

	log.Info("Runtime Pods", "size", len(runtimePods.Items), "runtime pods namelist", runtimePodNameList)

	// Record that the pod needs to be deleted.
	difference := runtimePodNameList.Difference(expectedPodNameList)

	// fill requiredPods with runtime if exists or expectedPod if not exists
	var requiredPods []corev1.Pod
	for _, pod := range expectedPods {
		if runtimePodNameList.Contains(pod.Name) {
			requiredPods = append(requiredPods, runtimePodMap[pod.Name])
		} else {
			requiredPods = append(requiredPods, pod)
		}
	}

	// Create the head node service.
	if needServicePodMap.Cardinality() > 0 {
		//podName := elem.(string)
		//svcConf := common.DefaultServiceConfig(*instance, podName)
		rayPodSvc := &instance.Spec.HeadService
		// Use a headless service, meaning that the DNS record for the service will
		// point directly to the head node pod's IP address.
		rayPodSvc.Spec.ClusterIP = corev1.ClusterIPNone

		blockOwnerDeletion := true
		ownerReference := metav1.OwnerReference{
			APIVersion:         instance.APIVersion,
			Kind:               instance.Kind,
			Name:               instance.Name,
			UID:                instance.UID,
			BlockOwnerDeletion: &blockOwnerDeletion,
		}
		rayPodSvc.OwnerReferences = append(rayPodSvc.OwnerReferences, ownerReference)
		if errSvc := r.Create(context.TODO(), rayPodSvc); errSvc != nil {
			if errors.IsAlreadyExists(errSvc) {
				log.Info("Pod service already exist,no need to create")
			} else {
				log.Error(errSvc, "Pod Service create error!", "Pod.Service.Error", errSvc)
				return reconcile.Result{}, errSvc
			}
		} else {
			log.Info("Pod Service created successfully", "service name", rayPodSvc.Name)
		}
		count := 0
		endPoint := &corev1.Endpoints{}
		for {
			if errEp := r.Get(context.TODO(), svcName, endPoint); errEp != nil {
				if !errors.IsNotFound(errEp) {
					log.Error(errEp, "getting endpoint error!", "Endpoint Error", errEp)
					break
				} else {
					if count < 2 {
						log.Info("waiting for endpoint ...")
						time.Sleep(2)
						count++
					} else {
						break
					}
				}
			}
			if len(endPoint.Subsets) != 0 {
				break
			} else {
				if count < 2 {
					log.Info("waiting for endpoint ...")
					time.Sleep(2)
					count++
				} else {
					break
				}
			}
		}
	}

	// Check if each pod exists and if not, create it.
	for i, replica := range requiredPods {
		if !utils.IsCreated(&replica) {
			log.Info("Creating pod", "index", i, "create pod", replica.Name)
			if err := r.Create(context.TODO(), &replica); err != nil {
				if errors.IsAlreadyExists(err) {
					log.Info("Creating pod", "Pod already exists", replica.Name)
				} else {
					return reconcile.Result{}, err
				}
			}
			//if this is a head, we want to wait until it is ready before creating the workers.
			if value, ok := replica.Labels["rayNodeType"]; ok && value == string(rayiov1alpha1.HeadNode) {
				podIdentifier := types.NamespacedName{
					Name:      replica.Name,
					Namespace: replica.Namespace,
				}
				count := 2
				for {
					if errPod := r.Get(context.TODO(), podIdentifier, &replica); errPod != nil {
						if !errors.IsNotFound(errPod) {
							log.Error(errPod, "getting pod error!", "podIdentifier", podIdentifier)
							break
						} else {
							if count < 2 {
								log.Info("waiting for pod ...")
								time.Sleep(2)
								count++
							} else {
								break
							}
						}
					}
					if replica.Status.Phase != corev1.PodRunning {
						if count < 2 {
							log.Info("waiting for head pod to be ready", "pod:", replica.Name)
							time.Sleep(2)
							count++
						} else {
							break
						}
					}
				}
				//end of wait logic
			}

		}
	}

	// Delete pods if needed.
	if difference.Cardinality() > 0 {
		log.Info("difference", "pods", difference)
		for _, runtimePod := range runtimePods.Items {
			if difference.Contains(runtimePod.Name) {
				log.Info("Deleting pod", "namespace", runtimePod.Namespace, "name", runtimePod.Name)
				if err := r.Delete(context.TODO(), &runtimePod); err != nil {
					return reconcile.Result{}, err
				}
				if strings.EqualFold(runtimePod.Labels[common.ClusterPodType], common.Head) {
					//svcConf := common.DefaultServiceConfig(*instance, runtimePod.Name)
					raySvcHead := &instance.Spec.HeadService
					log.Info("delete head service", "headName", runtimePod.Name)
					if err := r.Delete(context.TODO(), raySvcHead); err != nil {
						return reconcile.Result{}, err
					}
				}
			}
		}
	}

	return reconcile.Result{}, nil
}

// Build head instance pod(s).
func (r *RayClusterReconciler) buildHeadPods(instance *rayiov1alpha1.RayCluster, svcName types.NamespacedName) []corev1.Pod {
	var pods []corev1.Pod

	for i := int32(0); i < *instance.Spec.HeadGroupSpec.Replicas; i++ {
		podType := rayiov1alpha1.HeadNode
		podName := strings.ToLower(instance.Name + common.DashSymbol + string(rayiov1alpha1.HeadNode) + common.DashSymbol + utils.FormatInt32(i))
		podConf := common.DefaultHeadPodConfig(instance, podType, podName)
		pod := common.BuildPod(podConf, rayiov1alpha1.HeadNode, instance.Spec.HeadGroupSpec.RayStartParams, svcName)
		// Set raycluster instance as the owner and controller
		if err := controllerutil.SetControllerReference(instance, pod, r.Scheme); err != nil {
			log.Error(err, "Failed to set controller reference for raycluster pod")
		}
		pods = append(pods, *pod)
	}

	return pods
}

// Build worker instance pods.
func (r *RayClusterReconciler) buildWorkerPods(instance *rayiov1alpha1.RayCluster, svcName types.NamespacedName) []corev1.Pod {
	var pods []corev1.Pod
	for _, worker := range instance.Spec.WorkerGroupsSpec {
		for i := int32(0); i < *worker.Replicas; i++ {
			podType := rayiov1alpha1.WorkerNode
			podName := instance.Name + common.DashSymbol + string(podType) + common.DashSymbol + worker.GroupName + common.DashSymbol + utils.FormatInt32(i)
			podConf := common.DefaultWorkerPodConfig(instance, &worker, podType, podName)
			pod := common.BuildPod(podConf, rayiov1alpha1.WorkerNode, worker.RayStartParams, svcName)
			// Set raycluster instance as the owner and controller
			if err := controllerutil.SetControllerReference(instance, pod, r.Scheme); err != nil {
				log.Error(err, "Failed to set controller reference for raycluster pod")
			}
			pods = append(pods, *pod)
		}
	}

	return pods
}

// SetupWithManager builds the reconciler.
func (r *RayClusterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&rayiov1alpha1.RayCluster{}).Named("ray-operator").
		Watches(&source.Kind{Type: &corev1.Pod{}}, &handler.EnqueueRequestForOwner{
			IsController: true,
			OwnerType:    &rayiov1alpha1.RayCluster{},
		}).
		Complete(r)
}
