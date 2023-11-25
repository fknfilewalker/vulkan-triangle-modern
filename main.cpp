//#include <vulkan/vulkan_raii.hpp>
#include <vector>
import vulkan_hpp;

class App {
    
};

int main(int argc, char *argv[])
{
    vk::raii::Context context;
    
    vk::ApplicationInfo applicationInfo;
   /* applicationInfo.apiVersion = VK_VERSION_1_3;
    
    std::vector<const char*> extensions;
#ifdef __APPLE__
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif
    
    vk::InstanceCreateInfo instanceCreateInfo;
    instanceCreateInfo.setPApplicationInfo(&applicationInfo);
    instanceCreateInfo.setPEnabledExtensionNames(extensions);
#ifdef __APPLE__
    instanceCreateInfo.setFlags(vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR);
#endif
    vk::raii::Instance instance( context, instanceCreateInfo);*/
    
    return 0;
}
