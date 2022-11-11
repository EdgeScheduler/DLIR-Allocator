#ifndef __VALUEINFO_H__
#define __VALUEINFO_H__

#include <vector>
#include <string>
#include <ostream>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include "Common/Serializability.h"

namespace OnnxValueType
{
    std::string OnnxTypeToString(const ONNXTensorElementDataType type);
    auto OnnxTypeToCxxObject(const ONNXTensorElementDataType type);
    ONNXTensorElementDataType StringToOnnxType(const std::string str);
}

class ValueInfo : public virtual Serializability
{
public:
    ValueInfo(){};
    ValueInfo(const nlohmann::json &json);
    ValueInfo(const std::string &name, const std::vector<int64_t> &shape, const ONNXTensorElementDataType &type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ValueInfo(const char *name, const std::vector<int64_t> &shape, const ONNXTensorElementDataType &type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    /// @brief serial to json object
    /// @return one nlohmann::json object. if value not valid, it will be nullptr.
    virtual nlohmann::json ToJson() const;

    /// @brief load class to json
    /// @param nlohmann::json  object
    virtual void LoadFromJson(const nlohmann::json &json);

    /// @brief get input/output name in model
    /// @return name
    const std::string &GetName() const;

    /// @brief set input/output name in model
    /// @return name
    void SetName(std::string name);

    /// @brief get get input/output shapes in model
    /// @return shapes
    const std::vector<int64_t> &GetShape() const;

    /// @brief get total value count with this shape
    /// @return
    const std::int64_t &GetDataCount() const;

    /// @brief get get input/output dim-size in model
    /// @return dim size
    std::size_t GetDimSize() const;

    /// @brief get raw element data-type
    /// @return ONNXTensorElementDataType
    const ONNXTensorElementDataType &GetType() const;

    /// @brief get element data-type
    /// @return std::string
    std::string GetTypeString() const;

    friend std::ostream &operator<<(std::ostream &out, const ValueInfo &value);

private:
    void UpdateCount();

    int64_t dataCount;
    std::vector<int64_t> shape;
    std::string name;
    ONNXTensorElementDataType type;
};

#endif // __VALUEINFO_H__