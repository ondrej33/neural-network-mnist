#include <cassert>
#include "lingebra.hpp"


double squared_error(DoubleVec outputs, DoubleVec expected)
{
    assert(outputs.size() == expected.size());
    return (outputs - expected) * (outputs - expected) / outputs.size();
}