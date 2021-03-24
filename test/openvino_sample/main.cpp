#include <chrono>
#include <inference_engine.hpp>
#include "common.hpp"

using namespace std::chrono;
using namespace InferenceEngine;

int main(int argc, char **argv) {

    //std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;
    //showAvailableDevices();
    if(argc < 4){
        std::cout << "command line:" << std::endl;
        std::cout << "  isr input_image output_image sr_model" << std::endl;
        return -1;
    }

    
 
    

    char *device = "GPU";
    char *input_name = argv[1];
    char *output_name = argv[2];
    char *input_model = argv[3];//"/home/aslan/workspace/backup/openvino_model/rrdn/saved_model.xml";

    InferenceEngine::Core ie;
    // --------------------------- 1. Load inference engine -------------------------------------
    std::cout << ie.GetVersions( device ) << std::endl;

    // 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
    
    CNNNetwork network = ie.ReadNetwork(input_model);
    if (network.getOutputsInfo().size() != 1) throw std::logic_error("Sample supports topologies with 1 output only");
    if (network.getInputsInfo().size() != 1) throw std::logic_error("Sample supports topologies with 1 input only");
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 3. Configure input & output ---------------------------------------------
    // --------------------------- Prepare input blobs -----------------------------------------------------
     /** Taking information about all topology inputs **/
    InputsDataMap inputInfo(network.getInputsInfo());
    auto inputInfoItem = *inputInfo.begin();
    /** Specifying the precision of input data.
      * This should be called before load of the network to the device **/
    inputInfoItem.second->setPrecision(Precision::FP32);
    //inputInfoItem.second->setLayout(InferenceEngine::Layout::NCHW);
    //inputInfoItem.second->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::RGB);

    OutputsDataMap outputInfo(network.getOutputsInfo());
    // BlobMap outputBlobs;
    std::string firstOutputName;

    for (auto & item : outputInfo) {
        if (firstOutputName.empty()) {
            firstOutputName = item.first;
        }
        DataPtr outputData = item.second;
        if (!outputData) {
            throw std::logic_error("output data pointer is not valid");
        }

        item.second->setPrecision(Precision::FP32);
        //item.second->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::RGB);
    }


    // --------------------------- 4. Loading model to the device ------------------------------------------
    ExecutableNetwork executable_network = ie.LoadNetwork(network, device);

    // --------------------------- 5. Create infer request -------------------------------------------------
    std::cout << "Create infer request" << std::endl;
    InferRequest infer_request = executable_network.CreateInferRequest();
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 6. Prepare input --------------------------------------------------------
    //Load raw RGB 540 * 960 * 3    

    for (const auto & item : inputInfo) {
        MemoryBlob::Ptr minput = as<MemoryBlob>(infer_request.GetBlob(item.first));
        if (!minput) {
            std::cout << "We expect input blob to be inherited from MemoryBlob, " <<
                "but by fact we were not able to cast it to MemoryBlob" << std::endl;
            return 1;
        }
        // locked memory holder should be alive all time while access to its buffer happens
        auto ilmHolder = minput->wmap();

        /** Filling input tensor with images. First b channel, then g and r channels **/
        size_t num_channels = minput->getTensorDesc().getDims()[1];
        size_t image_size = minput->getTensorDesc().getDims()[3] * minput->getTensorDesc().getDims()[2];
        //for( int i  = 0; i < 4; i ++) std::cout << "DIM " << minput->getTensorDesc().getDims()[i] << std::endl;

        auto data = ilmHolder.as<PrecisionTrait<Precision::FP32>::value_type *>();
        if (data == nullptr)
            throw std::runtime_error("Input blob has not allocated buffer");
        
        std::ifstream input(input_name, std::ios::binary);
        if (!input) {
            std::cout << "Failed to open " << input_name << std::endl;
            return -1;
        }

        //input range [0, 1]
        unsigned char raw[540*960*3];
        input.read((char*)raw, image_size * num_channels);
        input.close();
        
        for(int pix = 0; pix < image_size; pix ++ )
            for(int ch = 0; ch < num_channels; ch ++)
                data[ch * image_size + pix] = raw[pix*num_channels + ch]/255.0;   
        
        /*
        for(int i = 0; i < 960; i ++)
            printf(" %f ", data[i]);
        printf("check input\n");
        */
    }
    

    // --------------------------- 7. Do inference ---------------------------------------------------------
    std::cout << "Start inference" << std::endl;
    high_resolution_clock::time_point start = high_resolution_clock::now();

    infer_request.Infer();

    high_resolution_clock::time_point now = high_resolution_clock::now();
    duration<double, std::milli> time_span = now - start;
    std::cout << "Infer took " << time_span.count() << " milliseconds.";
    std::cout << std::endl;

     // --------------------------- 8. Process output -------------------------------------------------------
    MemoryBlob::CPtr moutput = as<MemoryBlob>(infer_request.GetBlob(firstOutputName));
    if (!moutput) {
        throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                "but by fact we were not able to cast it to MemoryBlob");
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto lmoHolder = moutput->rmap();
    const auto output_data = lmoHolder.as<const PrecisionTrait<Precision::FP32>::value_type *>();

    size_t num_images = moutput->getTensorDesc().getDims()[0];
    size_t num_channels = moutput->getTensorDesc().getDims()[1];
    size_t H = moutput->getTensorDesc().getDims()[2];
    size_t W = moutput->getTensorDesc().getDims()[3];
    size_t nPixels = W * H;

    std::cout << "Output size [N,C,H,W]: " << num_images << ", " << num_channels << ", " << H << ", " << W << std::endl;

    //
    unsigned char data_img[1080 * 1920 * 3];

    for (size_t i = 0; i < nPixels; i++) {
        int r = output_data[i] * 255;
        int g = output_data[i + nPixels] * 255;
        int b = output_data[i + 2 * nPixels] * 255;

        if( r < 0 ) r = 0;
        else if( r > 255 ) r = 255;

        if( g < 0 ) g = 0;
        else if( g > 255 ) g = 255;

        if( b < 0 ) b = 0;
        else if( b > 255 ) b = 255;

        data_img[i * num_channels] = r;
        data_img[i * num_channels + 1] = g;
        data_img[i * num_channels + 2] = b;

    }


     std::ofstream outfile (output_name, std::ofstream::binary);
     outfile.write ((const char*)data_img, nPixels * num_channels);
     outfile.close();
    /*
     for(int i = 0; i < 1920 * 3; i++)
        printf("%d ", data_img[i]);
    printf("\n");
    */
            

    return 0;
}