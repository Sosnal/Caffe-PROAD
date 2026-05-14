#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include <future>
#include <mutex>
namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    offset_() {
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
  size_ = this->layer_param_.data_param().size();
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
    }
  }
}

template <typename Dtype>
bool DataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}
Datum datum1,datum2,datum3;
int i;
std::mutex mtx;

template<typename Dtype>
void DataLayer<Dtype>::Next() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
  i++;
  std::lock_guard<std::mutex> lock(mtx);
  if(i==1){
      datum1.ParseFromString(cursor_->value());
  }else if(i==2){
      datum2.ParseFromString(cursor_->value());
  }else{
      datum3.ParseFromString(cursor_->value());
      i=0;
  }
}
template<typename Dtype>
void DataLayer<Dtype>::Next1() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}

double total_batch_time = 0;
double total_read_time = 0;
double total_trans_time = 0;
double total_reshape_time = 0;
double total_phase1_time = 0;
double total_check_time = 0;
int iteration_count = 0;
// This function is called on prefetch thread
template<typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  double check_time = 0;
  double reshape_time = 0;
  double phase1_time = 0;
  CPUTimer timer;
    
  timer.Start();
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_param().batch_size();
  check_time += timer.MicroSeconds();
    
  Datum datum;
  if(size_ > 50){
      for (int item_id = 0; item_id < batch_size; ++item_id) {
        //shared_ptr<db::Cursor> prev_cursor_ = cursor_;
      //  LOG(INFO) <<"yes";
        timer.Start();
       // while (Skip()) {
       //   Next();
      //  }
      //  LOG(INFO) << "Key= " << cursor_->key();
       // datum.ParseFromString(cursor_->value());
        if(item_id == 0){
            datum.ParseFromString(cursor_->value());
        }else{
            std::lock_guard<std::mutex> lock(mtx);
            if(i==1){
                datum=datum1;
            }else if(i==2){
                datum=datum2;
            }else{
                datum=datum3;
            }
        }
        read_time += timer.MicroSeconds();
    
        timer.Start(); 
        if (item_id == 0) {
          // Reshape according to the first datum of each batch
          // on single input batches allows for inputs of varying dimension.
          // Use data_transformer to infer the expected blob shape from datum.
          vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
          this->transformed_data_.Reshape(top_shape);
          // Reshape batch according to the batch_size.
          top_shape[0] = batch_size;
          batch->data_.Reshape(top_shape);
        }
        reshape_time += timer.MicroSeconds();

        timer.Start();
        std::future<void> next_future = std::async(std::launch::async, &DataLayer<Dtype>::Next, this);
        phase1_time += timer.MicroSeconds();
    
        // Apply data transformations (mirror, scale, crop...)
        timer.Start();
        int offset = batch->data_.offset(item_id);
        Dtype* top_data = batch->data_.mutable_cpu_data();
        this->transformed_data_.set_cpu_data(top_data + offset);
        this->data_transformer_->Transform(datum, &(this->transformed_data_));
        // Copy label.
        if (this->output_labels_) {
          Dtype* top_label = batch->label_.mutable_cpu_data();
          top_label[item_id] = datum.label();
        }
        trans_time += timer.MicroSeconds();
        //Next();
      }
  }else{
        for (int item_id = 0; item_id < batch_size; ++item_id) {
        //shared_ptr<db::Cursor> prev_cursor_ = cursor_;
     //   LOG(INFO) <<"no";
        timer.Start();
        while (Skip()) {
          Next();
        }
      //  LOG(INFO) << "Key= " << cursor_->key();
        datum.ParseFromString(cursor_->value());
        read_time += timer.MicroSeconds();
    
        timer.Start(); 
        if (item_id == 0) {
          // Reshape according to the first datum of each batch
          // on single input batches allows for inputs of varying dimension.
          // Use data_transformer to infer the expected blob shape from datum.
          vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
          this->transformed_data_.Reshape(top_shape);
          // Reshape batch according to the batch_size.
          top_shape[0] = batch_size;
          batch->data_.Reshape(top_shape);
        }
        reshape_time += timer.MicroSeconds();
    
        // Apply data transformations (mirror, scale, crop...)
        timer.Start();
        int offset = batch->data_.offset(item_id);
        Dtype* top_data = batch->data_.mutable_cpu_data();
        this->transformed_data_.set_cpu_data(top_data + offset);
        this->data_transformer_->Transform(datum, &(this->transformed_data_));
        // Copy label.
        if (this->output_labels_) {
          Dtype* top_label = batch->label_.mutable_cpu_data();
          top_label[item_id] = datum.label();
        }
        trans_time += timer.MicroSeconds();
            
        timer.Start();
        Next1();
        phase1_time += timer.MicroSeconds();
      }
  }
  timer.Stop();
  batch_timer.Stop();
  total_batch_time += batch_timer.MilliSeconds();
  total_read_time += read_time;
  total_trans_time += trans_time;
  total_phase1_time += phase1_time;
  total_check_time += check_time;
  total_reshape_time += reshape_time;
  iteration_count++;

  LOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  LOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
  LOG(INFO) << "phase1 time: " << phase1_time / 1000 << " ms.";
  LOG(INFO) << "check time: " << check_time / 1000 << " ms.";
  LOG(INFO) << "reshape time: " << reshape_time / 1000 << " ms.";
    
  LOG(INFO) << " ";

  LOG(INFO) << "total Prefetch batch: " << total_batch_time << " ms.";
  LOG(INFO) << "     total Read time: " << total_read_time / 1000 << " ms.";
  LOG(INFO) << "total Transform time: " << total_trans_time / 1000 << " ms.";
  LOG(INFO) << "   total phase1 time: " << total_phase1_time / 1000 << " ms.";
  LOG(INFO) << "    total check time: " << total_check_time / 1000 << " ms.";
  LOG(INFO) << "  total reshape time: " << total_reshape_time / 1000 << " ms.";
  LOG(INFO) << "     iteration_count: " << iteration_count;

}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
