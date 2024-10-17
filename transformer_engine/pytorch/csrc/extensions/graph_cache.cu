

#include "extensions.h"
#include "common.h"

static GraphCache graph_cache;

void GraphCache::insert(at::Tensor tensor){
    TORCH_CHECK(!this->graph_locked);
    this->cache.push_back(tensor);
}

at::Tensor GraphCache::retrieve(){
    TORCH_CHECK(this->graph_locked);
    TORCH_CHECK(this->cache_index < this->cache.size(), "tried retrieving out of bounds");

    at::Tensor ret = this->cache[graph_cache.cache_index];
    this->cache_index += 1;
    return at::from_blob(ret.data_ptr(), ret.sizes(), ret.options());
}

void set_graph_cached_locked(){
    if (!graph_cache.graph_locked){
        graph_cache.graph_locked = true;
    }
    else{
        std::cout << "resetting index at value " << graph_cache.cache_index << std::endl;
        TORCH_CHECK(graph_cache.cache_index == graph_cache.cache.size(), 
            "Cudagraph cache: index reset in middle of layer!");
    }
    graph_cache.cache_index = 0;
}


void set_capture_start(){
    graph_cache.graph_capturing = true;
}
void set_capture_end(){
    graph_cache.graph_capturing = false;
}
bool is_graph_capturing(){
    return graph_cache.graph_capturing;
}

at::Tensor empty_like_cached(at::Tensor tensor, at::TensorOptions options){
    if (!graph_cache.graph_locked){
        at::Tensor copy = at::empty_like(tensor, options);
        graph_cache.insert(copy);

        std::cout << tensor.device() << " | EMPTY_LIKE2 ALLOCATE with shape" << tensor.sizes() 
            << "into index" << graph_cache.cache.size() -1 << std::endl;

        return copy;
    }
    else{
        at::Tensor ret = graph_cache.retrieve();
        std::cout << tensor.device() << " | EMPTY_LIKE2 RETRIEVE from index"
            << " from index: " << graph_cache.cache_index -1
            << " shape: " << ret.sizes()
            << " options: " << ret.options() << std::endl;

        TORCH_CHECK(ret.sizes() == tensor.sizes(), "cudagraph cache: size mismatch");
        TORCH_CHECK(ret.dtype() == tensor.dtype(), "cudagraph cache: dtype mismatch");
        TORCH_CHECK(ret.device() == tensor.device(), "cudagraph cache: device mismatch");

        return ret;
    }
}


at::Tensor empty_like_cached(at::Tensor tensor){
    if (!graph_cache.graph_locked){
        at::Tensor copy = at::empty_like(tensor);
        graph_cache.insert(copy);

        std::cout << tensor.device() << " | EMPTY_LIKE ALLOCATE with shape" << tensor.sizes() 
            << "into index" << graph_cache.cache.size() -1 << std::endl;

        return copy;
    }
    else{
        at::Tensor ret = graph_cache.retrieve();
        std::cout << tensor.device() << " | EMPTY_LIKE RETRIEVE from index"
            << " from index: " << graph_cache.cache_index -1
            << " shape: " << ret.sizes()
            << " options: " << ret.options() << std::endl;

        TORCH_CHECK(ret.sizes() == tensor.sizes(), "cudagraph cache: size mismatch");
        TORCH_CHECK(ret.dtype() == tensor.dtype(), "cudagraph cache: dtype mismatch");
        TORCH_CHECK(ret.device() == tensor.device(), "cudagraph cache: device mismatch");

        return ret;
    }
}

at::Tensor empty_cached(at::IntArrayRef size, at::ScalarType dtype, int device_index){
    at::Device device(at::kCUDA, device_index);
    auto options = at::TensorOptions()
        .dtype(dtype)
        .device(device);
    return empty_cached(size, options);
}

at::Tensor empty_cached(at::IntArrayRef size, at::ScalarType dtype, at::Device device){
    auto options = at::TensorOptions()
        .dtype(dtype)
        .device(device);
    return empty_cached(size, options);
}

at::Tensor empty_cached(at::IntArrayRef size, at::TensorOptions options){
    if (!graph_cache.graph_locked){
        at::Tensor copy = at::empty(size, options);
        graph_cache.insert(copy);

        std::cout << options.device() << " | EMPTY_CACHE ALLOCATE from index "
            << "from index: " << graph_cache.cache.size() -1
            << " shape: " <<size
            << " options: " << options << std::endl;

        return copy;
    }
    else{
        at::Tensor ret = graph_cache.retrieve();

        std::cout << options.device() << " | EMPTY_CACHE RETRIEVE from index " 
            << graph_cache.cache_index - 1
            << " returned options: " << ret.options()
            << " requeste options: " << options << std::endl;

        TORCH_CHECK(ret.sizes() == size, "cudagraph cache: size mismatch");
        TORCH_CHECK(ret.dtype() == options.dtype(), "cudagraph cache: dtype mismatch");
        // TORCH_CHECK(ret.device() == options.device(), "cudagraph cache: device mismatch");

        return ret;

    }
}


