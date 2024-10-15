

#include "extensions.h"
#include "common.h"

static GraphCache graph_cache;

void GraphCache::insert(at::Tensor tensor){
    TORCH_CHECK(!this->graph_locked);
    this->cache.push_back(tensor);
}

at::Tensor GraphCache::retrieve(){
    TORCH_CHECK(this->graph_locked);
    int cache_size = this->cache.size();
    TORCH_CHECK(this->cache_index < cache_size, "tried retrieving out of bounds");

    at::Tensor ret = this->cache[graph_cache.cache_index];
    this->cache_index += 1;
    return ret;
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

std::map<std::string, at::Tensor> cache = {}; 
at::Tensor empty_like_cached(at::Tensor tensor){
    if (!graph_cache.graph_locked){
        at::Tensor copy = at::empty_like(tensor);
        graph_cache.insert(copy);

        std::cout << tensor.device() << " |empty_like_cached allocated with shape" << tensor.sizes() 
            << "into index" << graph_cache.cache.size() -1 << std::endl;

        return copy;
    }
    else{
        std::cout << tensor.device() << " |retrieving from index " << graph_cache.cache_index << std::endl;
    }

    at::Tensor ret = graph_cache.retrieve();
    TORCH_CHECK(ret.sizes() == tensor.sizes(), "cudagraph cache: size mismatch");
    TORCH_CHECK(ret.dtype() == tensor.dtype(), "cudagraph cache: dtype mismatch");
    TORCH_CHECK(ret.device() == tensor.device(), "cudagraph cache: device mismatch");

    return ret;

}

// at::Tensor empty_cached(at::Tensor tensor){

// }

