#include "../../include/GPUAllocator/ModeCheck.h"

bool ModelCheck(std::string &mode)
{
    bool defined_mode = false;

#ifdef OYST_MODE
    if (defined_mode)
    {
        mode = "error-model";
        return false;
    }
    defined_mode = true;
    mode = "OYST-mode";
#endif

#ifdef BNST_MODE
    if (defined_mode)
    {
        mode = "error-model";
        return false;
    }
    defined_mode = true;
    mode = "BNST-mode";
#endif

#ifdef FIFO_MODE
    if (defined_mode)
    {
        mode = "error-model";
        return false;
    }
    defined_mode = true;
    mode = "FILO-mode";
#endif

#ifdef PARALLER_MODE
    if (defined_mode)
    {
        mode = "error-model";
        return false;
    }
    defined_mode = true;
    mode = "PARALLEL-mode";
#endif

    if (!defined_mode)
    {
        defined_mode = true;
        mode = "DLIR-mode";
    }

    return true;
}
