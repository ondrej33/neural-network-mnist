#include <vector>
#include <utility>
#include <cassert>
#include <iomanip>
#include <iostream>


/**
 * Vector of floats with all necessary operators defined 
 */

class DoubleVec
{
    std::vector<float> _vec;

public:
    DoubleVec(): _vec() {}

    /* construct a zero DoubleVec of given dimension */
    explicit DoubleVec(int dimension): _vec(std::vector<float>(dimension)) {}

    /* construct a DoubleVec from given vector of floats */
    explicit DoubleVec(std::vector<float> &v): _vec(v) {}
    explicit DoubleVec(std::vector<float> &&v): _vec(std::move(v)) {}

    /* std::vector operations */
    int size() const { return _vec.size(); }
    bool empty() const { return _vec.empty(); }
    void push_back(float num) { _vec.push_back(num); }
    void pop_back(float num) { _vec.pop_back(); }
    void reserve(int n) { _vec.reserve(n); }

    /* Arithmetic operators that modify sthe object */

    DoubleVec& operator+=(const DoubleVec& other)
    {
        this->compareSizes(other);
        for (int i = 0; i < size(); ++i) {
            _vec[i] += other._vec[i];
        }
        return *this;
    }

    DoubleVec& operator-=(const DoubleVec& other)
    {
        this->compareSizes(other);
        for (int i = 0; i < size(); ++i) {
            _vec[i] -= other._vec[i];
        }
        return *this;
    }

    DoubleVec& operator*=(float scalar)
    {
        for (int i = 0; i < size(); ++i) {
            _vec[i] *= scalar;
        }
        return *this;
    }

    // having *= for dot product does not make sense

    DoubleVec& operator/=(float scalar)
    {
        for (int i = 0; i < size(); ++i) {
            _vec[i] /= scalar;
        }
        return *this;
    }

    /* Other arithmetic operators that dont modify given objects */

    friend DoubleVec operator-(DoubleVec vec);
    friend DoubleVec operator+(DoubleVec first, const DoubleVec& second);
    friend DoubleVec operator-(DoubleVec first, const DoubleVec& second);
    friend float operator*(const DoubleVec& first, const DoubleVec& second);

    friend DoubleVec operator*(DoubleVec vec, float scalar);
    friend DoubleVec operator*(float scalar, DoubleVec vec);
    friend DoubleVec operator/(DoubleVec vec, float scalar);
    friend DoubleVec operator/(float scalar, DoubleVec vec);

    /* Squares the inside values */
    friend DoubleVec square_inside(DoubleVec vec);

    /* adds scalar to every item in vector */
    friend DoubleVec add_scalar_to_all_items(DoubleVec vec, float d);

    /* Relational operators */
    bool operator==(const DoubleVec& other) const { return this->_vec == other._vec; }
    bool operator!=(const DoubleVec& other) const { return !(*this == other); }

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

    void compareSizes(const DoubleVec& other) const;
};

/* unary minus */
DoubleVec operator-(DoubleVec vec)
{   
    for (int i = 0; i < vec._vec.size(); ++i) {
        vec[i] *= -1;
    }
    return vec;
}

/* matrix addition and subtraction */
DoubleVec operator+(DoubleVec first, const DoubleVec& second)
{ return first += second; }

DoubleVec operator-(DoubleVec first, const DoubleVec& second)
{ return first -= second; }

/* dot product */
float operator*(const DoubleVec& first, const DoubleVec& second)
{
    first.compareSizes(second);
    float result = 0;
    for (int i = 0; i < first.size(); ++i) {
        result += first[i] * second[i];
    }
    return result;
}

/* Multiplication and division by scalar */
DoubleVec operator*(DoubleVec vec, float scalar)
{ return vec *= scalar; }

DoubleVec operator*(float scalar, DoubleVec vec)
{ return vec *= scalar; }

DoubleVec operator/(DoubleVec vec, float scalar)
{ return vec /= scalar; }

DoubleVec operator/(float scalar, DoubleVec vec)
{ return vec /= scalar; }

/* Squares the inside values */
DoubleVec square_inside(DoubleVec vec)
{
    for (int i = 0; i < vec.size(); ++i) {
        vec[i] *= vec[i];
    }
    return vec;
}

/* Compute sqrt of the inside values */
DoubleVec sqrt_inside(DoubleVec vec)
{
    for (int i = 0; i < vec.size(); ++i) {
        vec[i] = std::sqrt(vec[i]);
    }
    return vec;
}

/* adds scalar to every item in vector */
DoubleVec add_scalar_to_all_items(DoubleVec vec, float d)
{      
    for (int i = 0; i < vec.size(); ++i) {
        vec[i] += d;
    }
    return vec;
}

/* divides vectors by items */
DoubleVec divide_by_items(DoubleVec first, const DoubleVec& second)
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
void DoubleVec::compareSizes(const DoubleVec& other) const {  
    #ifdef DEBUG
    if (size() != other.size()) {
        throw std::length_error("Vectors have different length");
    }    
    #endif
}


/**
 * Matrix of floats with all necessary operators defined 
 */

class DoubleMat
{
    std::vector<DoubleVec> _matrix_rows;
public:
    /* construct a zero matrix */
    explicit DoubleMat(int rows, int columns) : _matrix_rows(rows, DoubleVec(columns)) {}

    /* construct a DoubleMat from given vectors of DoubleVec */
    explicit DoubleMat(const std::vector<DoubleVec> &rows) : _matrix_rows(rows) {}
    explicit DoubleMat(const std::vector<DoubleVec> &&rows) : _matrix_rows(std::move(rows)) {}

    int row_num() const
    { return _matrix_rows.size(); }

    int col_num() const
    { return (row_num() == 0) ? 0 : _matrix_rows[0].size(); }

    bool same_dimensions(const DoubleMat &other) const
    { return col_num() == other.col_num() && row_num() == other.row_num(); }

    /* Direct access to the COPY of column and row vectors
     * The ‹n› is index from top (row) or left (column) */

    DoubleVec row( int n ) const
    { 
        assert(n >= 0 && n <= row_num() - 1);
        return _matrix_rows[n];
    }

    DoubleVec col( int n ) const
    {
        assert(n >= 0 && col_num() >= n + 1);

        DoubleVec column(row_num());
        for (int i = 0; i < row_num(); ++i) {
            column[i] = _matrix_rows[i][n];
        }
        return column;
    }

    /* relational operators */
    bool operator==(const DoubleMat &other) const
    { return _matrix_rows == other._matrix_rows; }

    bool operator!=(const DoubleMat &other) const
    { return _matrix_rows != other._matrix_rows; }

    /* constant / non-constant indexing to get or change the row vectors */
    DoubleVec operator[](int n) const
    { 
        assert(n >= 0 && n <= row_num() - 1);
        return _matrix_rows[n]; 
    }

    DoubleVec &operator[](int n)
    { 
        assert(n >= 0 && n <= row_num() - 1);
        return _matrix_rows[n]; 
    }

    /* matrix addition and subtraction */
    DoubleMat &operator+=(const DoubleMat &other)
    {        
        assert(same_dimensions(other));
        for (int i = 0; i < row_num(); ++i) {
            _matrix_rows[i] += other._matrix_rows[i];
        }
        return *this;
    }

    DoubleMat &operator-=(const DoubleMat &other)
    {
        assert(same_dimensions(other));
        for (int i = 0; i < row_num(); ++i) {
            _matrix_rows[i] -= other._matrix_rows[i];
        }
        return *this;
    }

    /* multiplying by a scalar */
    DoubleMat &operator*=(float scal)
    {
        for (int i = 0; i < row_num(); ++i) {
            _matrix_rows[i] *= scal;
        }
        return *this;
    }

    /* dividing by a scalar */
    DoubleMat &operator/=(float scal)
    {
        for (int i = 0; i < row_num(); ++i) {
            _matrix_rows[i] /= scal;
        }
        return *this;
    }

    // having *= for matrix product does not make sense

    friend DoubleMat square_inside(DoubleMat mat);

    friend DoubleMat add_scalar_to_all_items(DoubleMat mat, float d);

    /* adds given vector to all rows */
    DoubleMat &add_vec_to_all_rows(const DoubleVec &vec)
    {        
        assert(vec.size() == col_num());
        for (int i = 0; i < row_num(); ++i) {
            _matrix_rows[i] += vec;
        }
        return *this;
    }

    /* get a transpose of the matrix */
    DoubleMat transpose() const
    {
        auto transp = DoubleMat(col_num(), row_num());
        for (int i = 0; i < col_num(); ++i) {
            transp[i] = col(i);
        }
        return transp;
    }

    /* Computes dot product of row of first matrix and col of second matrix */
    friend float dot_row_col(const DoubleMat& first, const DoubleMat& second, int row, int col);

    /* Multiplies two matrices, the result is saved inside this matrix 
       Does not waste time or space */
    DoubleMat& save_multiplication(const DoubleMat& first, const DoubleMat& second)
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
    DoubleMat& save_multiplication_transponse_first(const DoubleMat& first, const DoubleMat& second)
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
    DoubleMat& save_multiplication_transponse_second(const DoubleMat& first, const DoubleMat& second)
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
    std::vector<DoubleVec>::iterator begin() { return _matrix_rows.begin(); }
    std::vector<DoubleVec>::iterator end() { return _matrix_rows.end(); }

    std::vector<DoubleVec>::const_iterator begin() const { return _matrix_rows.begin(); }
    std::vector<DoubleVec>::const_iterator end() const { return _matrix_rows.end(); }
};

/* Computes dot product of row of first matrix and col of second matrix */
float dot_row_col(const DoubleMat& first, const DoubleMat& second, int row, int col)
{
    assert(first.col_num() == second.row_num());
    float res = 0.;
    for (int i = 0; i < first.col_num(); ++i) {
        res += first[row][i] * second[i][col];
    }
    return res;
}

/* Computes dot product of row of matrix and vector */
float dot_row_vec(const DoubleMat& mat, const DoubleVec& vec, int row)
{
    assert(mat.col_num() == vec.size());
    float res = 0.;
    for (int i = 0; i < mat.col_num(); ++i) {
        res += mat[row][i] * vec[i];
    }
    return res;
}

/* Computes dot product of vector and col of matrix */
float dot_vec_col(const DoubleVec& vec, const DoubleMat& mat, int col)
{
    assert(vec.size() == mat.row_num());
    float res = 0.;
    for (int i = 0; i < vec.size(); ++i) {
        res += vec[i] * mat[i][col];
    }
    return res;
}

/* matrix addition and subtraction */
DoubleMat operator+(DoubleMat x, const DoubleMat &y)
{ return x += y; }

DoubleMat operator-(DoubleMat x, const DoubleMat &y)
{ return x -= y; }

/* multiplication of a matrix by a scalar - both sides */
DoubleMat operator*(DoubleMat m, float scalar)
{ return m *= scalar; }

DoubleMat operator*(float scalar, DoubleMat m)
{ return m *= scalar; }

/* division of a matrix by a scalar - both sides */
DoubleMat operator/(DoubleMat m, float scalar)
{ return m /= scalar; }

DoubleMat operator/(float scalar, DoubleMat m)
{ return m /= scalar; }

/* multiplication of a vector by a matrix - both sides 
 * if vector is first param, it is considered as [1, n] matrix
 * if vector is second param, it is considered as [n,1] matrix */
// TODO optimize
DoubleVec operator*(DoubleVec v, const DoubleMat &m)
{ 
    assert(m.row_num() == v.size());
    auto result = DoubleVec(m.col_num());

    for (int i = 0; i < m.col_num(); ++i) {
        result[i] = v * m.col(i);
    }
    return result;
}

// TODO optimize
DoubleVec operator*(const DoubleMat &m, DoubleVec v)
{ 
    assert(m.col_num() == v.size());
    auto result = DoubleVec(m.row_num());

    for (int i = 0; i < m.row_num(); ++i) {
        result[i] = m.row(i) * v;
    }
    return result;
}

/* multiplication of compatible matrices */
// TODO optimize
DoubleMat operator*(const DoubleMat &first, const DoubleMat &second)
{ 
    assert(first.col_num() == second.row_num());
    auto result = DoubleMat(first.row_num(), second.col_num());

    for (int i = 0; i < first.row_num(); ++i) {
        for (int j = 0; j < second.col_num(); ++j) {
            result[i][j] = first.row(i) * second.col(j);
        }
    }
    return result;
}

/* Squares the inside values */
DoubleMat square_inside(DoubleMat mat)
{
    for (int i = 0; i < mat.row_num(); ++i) {
        mat[i] = square_inside(mat[i]);
    }
    return mat;
}

/* Compute sqrt of the inside values */
DoubleMat sqrt_inside(DoubleMat mat)
{
    for (int i = 0; i < mat.row_num(); ++i) {
        mat[i] = sqrt_inside(mat[i]);
    }
    return mat;
}

/* adds given vector to all rows */
DoubleMat add_scalar_to_all_items(DoubleMat mat, float d)
{        
    for (int i = 0; i < mat.row_num(); ++i) {
        mat[i] = std::move(add_scalar_to_all_items(mat[i], d));
    }
    return mat;
}

/* divides matrices by items */
DoubleMat divide_by_items(DoubleMat first, const DoubleMat& second)
{      
    assert(first.col_num() == second.col_num() && first.row_num() == second.row_num());
    for (int i = 0; i < first.row_num(); ++i) {
        first[i] = std::move(divide_by_items(first[i], second[i]));
    }
    return first;
}

/* printing a matrix, float printed to 4 decimal places */
inline void print_matrix(const DoubleMat &m)
{
    std::cout << std::setprecision(4) << std::fixed;

    for (int i = 0; i < m.row_num(); ++i) {
       for (int j = 0; j < m.col_num(); ++j) {
            std::cout << m[i][j] << " ";
        }
        std::cout << "\n";
    }
}
