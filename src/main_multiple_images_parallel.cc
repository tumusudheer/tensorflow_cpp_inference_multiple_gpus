/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// To use it, compile and then run in a working directory with the
// learning/brain/tutorials/label_image/data/ folder below it, and you should
// see the top five labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//
// The googlenet_graph.pb file included by default is created from Inception.
//
// Note that, for GIF inputs, to reuse existing code, only single-frame ones
// are supported.

#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include <sys/time.h>
#include <pthread.h>
//#include "cintel_inception.h"


// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

#define MAX_THREADS 64
int nthreads = 2;
pthread_t threads[MAX_THREADS];
char **global_images;
int global_number_of_images;

std::shared_ptr<tensorflow::Session> global_session_00; // GPU for inception for thread 0
std::shared_ptr<tensorflow::Session> global_session_01; // GPU for inception for thread 1

std::shared_ptr<tensorflow::Session> global_session_10; // for image ops on CPU such as reading & resizing images

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}


static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<string>()() = string(data);
  return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status LoadImageResizeGraph(std::shared_ptr<tensorflow::Session>* session, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std) {
  string output_name = "normalized";
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  // use a placeholder to read input data
  auto file_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
      {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph_def;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph_def));

  tensorflow::SessionOptions session_options;
  session_options.config.mutable_gpu_options()->set_visible_device_list("0");
  session_options.config.mutable_gpu_options()->set_allow_growth(true);
  session_options.config.set_allow_soft_placement(true);
  session->reset(tensorflow::NewSession(session_options));
  tensorflow::graph::SetDefaultDevice("/cpu:0", &graph_def);
  TF_RETURN_IF_ERROR((*session)->Create(graph_def));
  return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::shared_ptr<tensorflow::Session>* session, int i_threadid) {
  tensorflow::GraphDef graph_def;
  printf(".. IN load graph %d\n", i_threadid);
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  tensorflow::SessionOptions session_options;
  if(i_threadid == 0) {
    session_options.config.mutable_gpu_options()->set_visible_device_list("0");
    tensorflow::graph::SetDefaultDevice("/device:GPU:0", &graph_def);
  } else if(i_threadid == 1) {
    session_options.config.mutable_gpu_options()->set_visible_device_list("0,1");
    tensorflow::graph::SetDefaultDevice("/device:GPU:1", &graph_def);
  }
  session_options.config.mutable_gpu_options()->set_allow_growth(true);
  session_options.config.set_allow_soft_placement(true);
  session->reset(tensorflow::NewSession(session_options));

  //tensorflow::graph::SetDefaultDevice("/cpu:0", &graph_def);
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

int Initialize(const string& graph_file_name, int i_nthreads) {
  int32 input_width = 299;
  int32 input_height = 299;
  float input_mean = 0;
  float input_std = 255;
  nthreads = i_nthreads;

  // Initialize all sessions;
  for(int i = 0; i < nthreads; i++) {

    int threadid = i;
    if(threadid == 0) {
      LoadGraph(graph_file_name, &global_session_00, 0);
    } else if (threadid == 1) {
      LoadGraph(graph_file_name, &global_session_10, 1);
    }
  }

  LoadImageResizeGraph(&global_session_01, input_height, input_width, input_mean, input_std);
}

void *Perform_Computation_Parallel(void *arg1)
{

  
  string input_layer = "input";
  string output_layer = "InceptionV3/Predictions/Reshape_1";


  int threadid = (int)(((size_t)(arg1)));
  printf("threadid %d\n", threadid);

  std::shared_ptr<tensorflow::Session> session;
  if(threadid == 0) {
    session = global_session_00;
  } else if(threadid == 1) {
    session = global_session_10;
  }

  char *image_path = (char *)malloc(1000000);
  for(int r = threadid; r < global_number_of_images; r += nthreads)
  {
    sprintf(image_path, "%s", global_images[r]);
    printf("%s --------> %d \n", image_path, threadid);

    std::string image_file(image_path);
    std::vector<Tensor> resized_tensors;

    // read file_name into a tensor named input
    Tensor input_image(tensorflow::DT_STRING, tensorflow::TensorShape());
    
    ReadEntireFile(tensorflow::Env::Default(), image_file, &input_image);

    Status run_resize_status = global_session_01->Run({{"input", input_image}},
                                      {"normalized"}, {}, &resized_tensors);

    if (!run_resize_status.ok()) {
      LOG(ERROR) << "Running resize graph failed: " << run_resize_status;
    } 
    // else if(run_resize_status.ok()){
    //   std::cout << "successfully resized image "<< image_file << "\n";
    // }

    // Status read_tensor_status =
    //         ReadTensorFromImageFile(image_file, input_height, input_width, input_mean,
    //                                 input_std, &resized_tensors);
    // if (!read_tensor_status.ok()) {
    //   LOG(ERROR) << read_tensor_status;
    //   //return -1;
    // }
    const Tensor& resized_tensor = resized_tensors[0];


    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{input_layer, resized_tensor}},
                                     {output_layer}, {}, &outputs);
    std::cout << "MOdel ran succesfully : output size " << outputs.size() << "\n";
    if (!run_status.ok()) {
      LOG(ERROR) << "Running model failed: " << run_status;
      //return -1;
    }
  }
  free(image_path);
}

void Perform_Computation(void)
{
    for(long long int i = 1; i < nthreads; i++) pthread_create(&threads[i], NULL, Perform_Computation_Parallel, (void *)(i));
    Perform_Computation_Parallel(0);
    for(long long int i = 1; i < nthreads; i++) pthread_join(threads[i], NULL);
}

int main(int argc, char* argv[]) {
  //printf("Hello World!!!\n");

  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than inception_v3, then you'll need to update these.
  string image = "./data/grace_hopper.jpg";
  string graph =
      "./data/inception_v3_2016_08_28_frozen.pb";
  string labels =
      "./data/imagenet_slim_labels.txt";
  int32 input_width = 299;
  int32 input_height = 299;
  float input_mean = 0;
  float input_std = 255;
  string input_layer = "input";
  string output_layer = "InceptionV3/Predictions/Reshape_1";
  bool self_test = false;
  string root_dir = "";
  string test_file = "./data/images/test_images_1000.txt";
  string out_file = "./data/results/test_results_1000.txt";

  //nthreads = 2;
  int ret_init_val = Initialize(graph, nthreads);
  std::cout << " Initialization done --------------------------\n";
  FILE *fp = fopen(test_file.c_str(), "r");
  int nol = 0;


  char *line = (char *)malloc(2048);

  while (!feof(fp))
  {
    line[0] = '\0';
    fgets(line, 2048, fp);
    if (line[0] == '\0')  break;
    nol++;
  }
  fclose(fp);

  global_number_of_images = nol;

  global_images = (char **)malloc(global_number_of_images * sizeof(char *));
  

  fp = fopen(test_file.c_str(), "r");
  for(int q = 0; q < global_number_of_images; q++)
  {
    line[0] = '\0';
    fgets(line, 2048, fp);
    line[strlen(line)-1] = '\0';
    global_images[q] = (char *)malloc(strlen(line)+10);
    sprintf(global_images[q], "%s", line);
  }
  fclose(fp);
  double t0 = elapsed();
  Perform_Computation();
  printf ("[%.3f s] Finished runnning %d images successfully\n", elapsed() - t0, nol);
  return 0;
}
