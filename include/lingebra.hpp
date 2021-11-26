#include <vector>
#include <utility>
#include <cassert>
#include <iomanip>
#include <iostream>


/**
 * Vector of floats with all necessary operators defined 
 */

class FloatVec
{
    std::vector<float> _vec;

public:
    FloatVec(): _vec() {}

    /* construct a zero FloatVec of given dimension */
    explicit FloatVec(int dimension): _vec(std::vector<float>(dimension)) {}

    /* construct a FloatVec from given vector of floats */
    explicit FloatVec(std::vector<float> &v): _vec(v) {}
    explicit FloatVec(std::vector<float> &&v): _vec(std::move(v)) {}

    /* std::vector operations */
    int size() const { return _vec.size(); }
    bool empty() const { return _vec.empty(); }
    void push_back(float num) { _vec.push_back(num); }
    void pop_back(float num) { _vec.pop_back(); }
    void reserve(int n) { _vec.reserve(n); }

    /* Arithmetic operators that modify sthe object */

    FloatVec& operator+=(const FloatVec& other)
    {
        this->compareSizes(other);
        for (int i = 0; i < size(); ++i) {
            _vec[i] += other._vec[i];
        }
        return *this;
    }

    FloatVec& operator-=(const FloatVec& other)
    {
        this->compareSizes(other);
        for (int i = 0; i < size(); ++i) {
            _vec[i] -= other._vec[i];
        }
        return *this;
    }

    FloatVec& operator*=(float scalar)
    {
        for (int i = 0; i < size(); ++i) {
            _vec[i] *= scalar;
        }
        return *this;
    }

    // having *= for dot product does not make sense

    FloatVec& operator/=(float scalar)
    {
        for (int i = 0; i < size(); ++i) {
            _vec[i] /= scalar;
        }
        return *this;
    }

    /* Other arithmetic operators that dont modify given objects */

    friend FloatVec operator-(FloatVec vec);
    friend FloatVec operator+(FloatVec first, const FloatVec& second);
    friend FloatVec operator-(FloatVec first, const FloatVec& second);
    friend float operator*(const FloatVec& first, const FloatVec& second);

    friend FloatVec operator*(FloatVec vec, float scalar);
    friend FloatVec operator*(float scalar, FloatVec vec);
    friend FloatVec operator/(FloatVec vec, float scalar);
    friend FloatVec operator/(float scalar, FloatVec vec);

    /* Squares the inside values */
    friend FloatVec square_inside(FloatVec vec);

    /* adds scalar to every item in vector */
    friend FloatVec add_scalar_to_all_items(FloatVec vec, float d);

    /* Relational operators */
    bool operator==(const FloatVec& other) const { return this->_vec == other._vec; }
    bool operator!=(const FloatVec& other) const { return !(*this == other); }

    /* Indexing and iterating */

    float operator[](int i) const 
    { 
        assert(i >= 0 && i <= size() - 1);
        return _vec[i]; 
    }

    float& operator[](int i) 
    { 
        assert(i >= 0 && i <= size() - 1);
        return _vec[i]; 
    }

    std::vector<float>::iterator begin() { return _vec.begin(); }
    std::vector<float>::iterator end() { return _vec.end(); }

    std::vector<float>::const_iterator begin() const { return _vec.begin(); }
    std::vector<float>::const_iterator end() const { return _vec.end(); }

    void compareSizes(const FloatVec& other) const;
};

/* unary minus */
FloatVec operator-(FloatVec vec)
{   
    for (int i = 0; i < vec._vec.size(); ++i) {
        vec[i] *= -1;
    }
    return vec;
}

/* matrix addition and subtraction */
FloatVec operator+(FloatVec first, const FloatVec& second)
{ return first += second; }

FloatVec operator-(FloatVec first, const FloatVec& second)
{ return first -= second; }

/* dot product */
float operator*(const FloatVec& first, const FloatVec& second)
{
    first.compareSizes(second);
    float result = 0;
    for (int i = 0; i < first.size(); ++i) {
        result += first[i] * second[i];
    }
    return result;
}

/* Multiplication and division by scalar */
FloatVec operator*(FloatVec vec, float scalar)
{ return vec *= scalar; }

FloatVec operator*(float scalar, FloatVec vec)
{ return vec *= scalar; }

FloatVec operator/(FloatVec vec, float scalar)
{ return vec /= scalar; }

FloatVec operator/(float scalar, FloatVec vec)
{ return vec /= scalar; }

/* Squares the inside values */
FloatVec square_inside(FloatVec vec)
{
    for (int i = 0; i < vec.size(); ++i) {
        vec[i] *= vec[i];
    }
    return vec;
}

/* Compute sqrt of the inside values */
FloatVec sqrt_inside(FloatVec vec)
{
    for (int i = 0; i < vec.size(); ++i) {
        vec[i] = std::sqrt(vec[i]);
    }
    return vec;
}

/* adds scalar to every item in vector */
FloatVec add_scalar_to_all_items(FloatVec vec, float d)
{      
    for (int i = 0; i < vec.size(); ++i) {
        vec[i] += d;
    }
    return vec;
}

/* divides vectors by items */
FloatVec divide_by_items(FloatVec first, const FloatVec& second)
{      
    assert(first.size() == second.size());
    for (int i = 0; i < first.size(); ++i) {
        first[i] /= second[i];
    }
    return first;
}

/**
 * @brief Check if the another vector is of the same length as ours
 *      Can be removed during production, just make this function empty and
 *      compiler will do the rest (Preferably by DEBUG macro )
 * @param other 
 */
void FloatVec::compareSizes(const FloatVec& other) const {  
    #ifdef DEBUG
    if (size() != other.size()) {
        throw std::length_error("Vectors have different length");
    }    
    #endif
}


/**
 * Matrix of floats with all necessary operators defined 
 */

class FloatMat
{
    std::vector<FloatVec> _matrix_rows;
public:
    /* construct a zero matrix */
    explicit FloatMat(int rows, int columns) : _matrix_rows(rows, FloatVec(columns)) {}

    /* construct a FloatMat from given vectors of FloatVec */
    explicit FloatMat(const std::vector<FloatVec> &rows) : _matrix_rows(rows) {}
    explicit FloatMat(const std::vector<FloatVec> &&rows) : _matrix_rows(std::move(rows)) {}

    /* "convert" one vector to matrix - either one-row matrix or one-column matrix */
    friend FloatMat row_mat_from_vec(const FloatVec &vec);
    friend FloatMat col_mat_from_vec(const FloatVec &vec);

    int row_num() const
    { return _matrix_rows.size(); }

    int col_num() const
    { return (row_num() == 0) ? 0 : _matrix_rows[0].size(); }

    bool same_dimensions(const FloatMat &other) const
    { return col_num() == other.col_num() && row_num() == other.row_num(); }

    /* Direct access to the COPY of column and row vectors
     * The ‹n› is index from top (row) or left (column) */

    FloatVec row( int n ) const
    { 
        assert(n >= 0 && n <= row_num() - 1);
        return _matrix_rows[n];
    }

    FloatVec col( int n ) const
    {
        assert(n >= 0 && col_num() >= n + 1);

        FloatVec column(row_num());
        for (int i = 0; i < row_num(); ++i) {
            column[i] = _matrix_rows[i][n];
        }
        return column;
    }

    /* relational operators */
    bool operator==(const FloatMat &other) const
    { return _matrix_rows == other._matrix_rows; }

    bool operator!=(const FloatMat &other) const
    { return _matrix_rows != other._matrix_rows; }

    /* constant / non-constant indexing to get or change the row vectors */
    FloatVec operator[](int n) const
    { 
        assert(n >= 0 && n <= row_num() - 1);
        return _matrix_rows[n]; 
    }

    FloatVec &operator[](int n)
    { 
        assert(n >= 0 && n <= row_num() - 1);
        return _matrix_rows[n]; 
    }

    /* matrix addition and subtraction */
    FloatMat &operator+=(const FloatMat &other)
    {        
        assert(same_dimensions(other));
        for (int i = 0; i < row_num(); ++i) {
            _matrix_rows[i] += other._matrix_rows[i];
        }
        return *this;
    }

    FloatMat &operator-=(const FloatMat &other)
    {
        assert(same_dimensions(other));
        for (int i = 0; i < row_num(); ++i) {
            _matrix_rows[i] -= other._matrix_rows[i];
        }
        return *this;
    }

    /* multiplying by a scalar */
    FloatMat &operator*=(float scal)
    {
        for (int i = 0; i < row_num(); ++i) {
            _matrix_rows[i] *= scal;
        }
        return *this;
    }

    /* dividing by a scalar */
    FloatMat &operator/=(float scal)
    {
        for (int i = 0; i < row_num(); ++i) {
            _matrix_rows[i] /= scal;
        }
        return *this;
    }

    // having *= for matrix product does not make sense

    friend FloatMat square_inside(FloatMat mat);

    friend FloatMat add_scalar_to_all_items(FloatMat mat, float d);

    /* adds given vector to all rows */
    FloatMat &add_vec_to_all_rows(const FloatVec &vec)
    {        
        assert(vec.size() == col_num());
        for (int i = 0; i < row_num(); ++i) {
            _matrix_rows[i] += vec;
        }
        return *this;
    }

    /* get a transpose of the matrix */
    FloatMat transpose() const
    {
        auto transp = FloatMat(col_num(), row_num());
        for (int i = 0; i < col_num(); ++i) {
            transp[i] = col(i);
        }
        return transp;
    }

    /* Computes dot product of row of first matrix and col of second matrix */
    friend float dot_row_col(const FloatMat& first, const FloatMat& second, int row, int col);

    /* Multiplies two matrices, the result is saved inside this matrix 
       Does not waste time or space */
    FloatMat& save_multiplication(const FloatMat& first, const FloatMat& second)
    {
        assert(first.col_num() == second.row_num());
        assert(first.row_num() == this->row_num() && second.col_num() == this->col_num());
        for (int i = 0; i < first.row_num(); ++i) {
            for (int j = 0; j < second.col_num(); ++j) {
                (*this)[i][j] = dot_row_col(first, second, i, j);
            }
        }
    }

    /* Multiplies TRANSPONED first matrix with second, the result is saved inside this matrix 
       Does not waste time or space */
    FloatMat& save_multiplication_transponse_first(const FloatMat& first, const FloatMat& second)
    {
        assert(first.row_num() == second.row_num());
        assert(first.col_num() == this->row_num() && second.col_num() == this->col_num());
        for (int i = 0; i < first.col_num(); ++i) {
            for (int j = 0; j < second.col_num(); ++j) {
                float res = 0.;
                for (int k = 0; k < first.row_num(); ++k) {
                    res += first[k][i] * second[k][j];
                }
                (*this)[i][j] = res;
            }
        }
        return *this;
    }

    /* Multiplies first matrix with TRANSPONED second, the result is saved inside this matrix 
       Does not waste time or space */
    FloatMat& save_multiplication_transponse_second(const FloatMat& first, const FloatMat& second)
    {
        assert(first.col_num() == second.col_num());
        assert(first.row_num() == this->row_num() && second.row_num() == this->col_num());
        for (int i = 0; i < first.row_num(); ++i) {
            for (int j = 0; j < second.row_num(); ++j) {
                float res = 0.;
                for (int k = 0; k < first.col_num(); ++k) {
                    res += first[i][k] * second[j][k];
                }
                (*this)[i][j] = res;
            }
        }
        return *this;
    }

    /* iterators */
    std::vector<FloatVec>::iterator begin() { return _matrix_rows.begin(); }
    std::vector<FloatVec>::iterator end() { return _matrix_rows.end(); }

    std::vector<FloatVec>::const_iterator begin() const { return _matrix_rows.begin(); }
    std::vector<FloatVec>::const_iterator end() const { return _matrix_rows.end(); }
};

/* "convert" one vector to matrix - either one-row matrix or one-column matrix */
FloatMat row_mat_from_vec(const FloatVec &vec)
{
    return FloatMat(std::vector<FloatVec>{vec});
}
FloatMat col_mat_from_vec(const FloatVec &vec)
{
    return FloatMat(std::vector<FloatVec>{vec}).transpose();
}

/* Computes dot product of row of first matrix and col of second matrix */
float dot_row_col(const FloatMat& first, const FloatMat& second, int row, int col)
{
    assert(first.col_num() == second.row_num());
    float res = 0.;
    for (int i = 0; i < first.col_num(); ++i) {
        res += first[row][i] * second[i][col];
    }
    return res;
}

/* Computes dot product of row of matrix and vector */
float dot_row_vec(const FloatMat& mat, const FloatVec& vec, int row)
{
    assert(mat.col_num() == vec.size());
    float res = 0.;
    for (int i = 0; i < mat.col_num(); ++i) {
        res += mat[row][i] * vec[i];
    }
    return res;
}

/* Computes dot product of vector and col of matrix */
float dot_vec_col(const FloatVec& vec, const FloatMat& mat, int col)
{
    assert(vec.size() == mat.row_num());
    float res = 0.;
    for (int i = 0; i < vec.size(); ++i) {
        res += vec[i] * mat[i][col];
    }
    return res;
}

/* matrix addition and subtraction */
FloatMat operator+(FloatMat x, const FloatMat &y)
{ return x += y; }

FloatMat operator-(FloatMat x, const FloatMat &y)
{ return x -= y; }

/* multiplication of a matrix by a scalar - both sides */
FloatMat operator*(FloatMat m, float scalar)
{ return m *= scalar; }

FloatMat operator*(float scalar, FloatMat m)
{ return m *= scalar; }

/* division of a matrix by a scalar - both sides */
FloatMat operator/(FloatMat m, float scalar)
{ return m /= scalar; }

FloatMat operator/(float scalar, FloatMat m)
{ return m /= scalar; }

/* multiplication of a vector by a matrix - both sides 
 * if vector is first param, it is considered as [1, n] matrix
 * if vector is second param, it is considered as [n,1] matrix */
// TODO optimize
FloatVec operator*(FloatVec v, const FloatMat &m)
{ 
    assert(m.row_num() == v.size());
    auto result = FloatVec(m.col_num());

    for (int i = 0; i < m.col_num(); ++i) {
        result[i] = v * m.col(i);
    }
    return result;
}

// TODO optimize
FloatVec operator*(const FloatMat &m, FloatVec v)
{ 
    assert(m.col_num() == v.size());
    auto result = FloatVec(m.row_num());

    for (int i = 0; i < m.row_num(); ++i) {
        result[i] = m.row(i) * v;
    }
    return result;
}

/* multiplication of compatible matrices */
// TODO optimize
FloatMat operator*(const FloatMat &first, const FloatMat &second)
{ 
    assert(first.col_num() == second.row_num());
    auto result = FloatMat(first.row_num(), second.col_num());

    for (int i = 0; i < first.row_num(); ++i) {
        for (int j = 0; j < second.col_num(); ++j) {
            result[i][j] = first.row(i) * second.col(j);
        }
    }
    return result;
}

/* Squares the inside values */
FloatMat square_inside(FloatMat mat)
{
    for (int i = 0; i < mat.row_num(); ++i) {
        mat[i] = square_inside(mat[i]);
    }
    return mat;
}

/* Compute sqrt of the inside values */
FloatMat sqrt_inside(FloatMat mat)
{
    for (int i = 0; i < mat.row_num(); ++i) {
        mat[i] = sqrt_inside(mat[i]);
    }
    return mat;
}

/* adds given vector to all rows */
FloatMat add_scalar_to_all_items(FloatMat mat, float d)
{        
    for (int i = 0; i < mat.row_num(); ++i) {
        mat[i] = std::move(add_scalar_to_all_items(mat[i], d));
    }
    return mat;
}

/* divides matrices by items */
FloatMat divide_by_items(FloatMat first, const FloatMat& second)
{      
    assert(first.col_num() == second.col_num() && first.row_num() == second.row_num());
    for (int i = 0; i < first.row_num(); ++i) {
        first[i] = std::move(divide_by_items(first[i], second[i]));
    }
    return first;
}

/* printing a matrix, float printed to 4 decimal places */
inline void print_matrix(const FloatMat &m)
{
    std::cout << std::setprecision(4) << std::fixed;

    for (int i = 0; i < m.row_num(); ++i) {
       for (int j = 0; j < m.col_num(); ++j) {
            std::cout << m[i][j] << " ";
        }
        std::cout << "\n";
    }
}
