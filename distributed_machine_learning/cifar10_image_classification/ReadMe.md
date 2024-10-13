

# Kubeflow Pipeline Setup with KServe and Prometheus Monitoring

This guide covers the installation and setup of Kubeflow Pipeline (version 2.2.0), KServe (version 0.13), and Prometheus monitoring on a Kubernetes cluster using Docker Desktop.

## Prerequisites
- **Docker Desktop**: Version 4.31.0
- **Kubernetes**: Version 1.29.1 (configured via Docker Desktop)
- **Kubeflow**: Installed (Standalone Deployment) Version 2.2.0

## 1. Kubeflow Installation
To install Kubeflow, execute the following commands:

```bash
export PIPELINE_VERSION=2.2.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"
```

## 2. KServe Installation for Deployment
To install KServe version 0.13, follow the instructions on the official KServe website: [KServe Kubernetes Deployment Guide](https://kserve.github.io/website/master/admin/kubernetes_deployment/)
I have followed the Raw Deployment method for our usecase.

## 3. Installing Additional Services
Install the following services for service account usage, ingress setup using Istio, Minio credential setup, resource binding, cluster roles for KServe, and Pushgateway for monitoring Prometheus.

Apply the respective YAML files using the following command:
```bash
kubectl apply -f yaml_file_name.yaml
```
Replace `yaml_file_name.yaml` with the corresponding file name from the folder:

## 4. Port Forwarding for Services
Before working with the pipeline, forward some Kubeflow services using the following commands:

### 4.1. Kubeflow UI
```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

### 4.2. Minio Service
For making an S3-like structure for local deployment:
```bash
kubectl port-forward svc/minio-service -n kubeflow 9000:9000
```

### 4.3. Pushgateway Service
For monitoring:
```bash
kubectl port-forward pushgateway-podname -n kubeflow 9090:9091
```
Replace `pushgateway-podname` with the actual pod name.

## 5. Prometheus Installation
To install Prometheus using Kubernetes, follow these steps:

### 5.1. Install Prometheus Operator
The Prometheus Operator simplifies Prometheus deployment and management.

```bash
kubectl create namespace monitoring
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml
```

### 5.2. Install Prometheus Stack (Prometheus, Alertmanager, Grafana)
You can install the complete Prometheus stack using Helm. First, ensure Helm is installed on your system.

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring
```

### 5.3. Port Forward Prometheus and Grafana
To access Prometheus and Grafana, forward the ports:

- **Prometheus**:
    ```bash
    kubectl port-forward svc/prometheus-kube-prometheus-prometheus -n monitoring 9090:9090
    ```
- **Grafana**:
    ```bash
    kubectl port-forward svc/prometheus-grafana -n monitoring 3000:80
    ```

Now you can access Prometheus at `http://localhost:9090` and Grafana at `http://localhost:3000`.

### 5.4. Add Prometheus as a Data Source in Grafana
1. Open Grafana in your browser: `http://localhost:3000`.
2. Go to **Configuration** -> **Data Sources**.
3. Click on **Add data source**.
4. Select **Prometheus**.
5. Set the URL to `http://prometheus-kube-prometheus-prometheus.monitoring.svc:9090`.
6. Click **Save & Test**.

Your Prometheus stack is now fully integrated and ready for monitoring!

