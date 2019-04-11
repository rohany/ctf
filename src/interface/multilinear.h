#ifndef __MULTILINEAR_H__
#define __MULTILINEAR_H__


namespace CTF {
  template<typename dtype>
  class Tensor;
  
  class Idx_Tensor;

  /**
   * \brief Compute updates to entries in tensor A based on matrices or vectors in mat_list (tensor times tensor products).
   *        Takes products of entries of A with multilinear dot products of columns of given matrices.
   *        For ndim=3 and mat_list=[X,Y,Z], this operation is equivalent to einsum("ijk,ia,ja,ka->ijk",A,X,Y,Z).
   *        FIXME: ignores semiring and just multiplies
   * \param[in] num_ops number of operands (matrices or vectors)
   * \param[in] modes modes on which to apply the operands
   * \param[in] mat_list where ith tensor is n_i-by-k or k-by-n_i matrix or vector of dim n_i where n_i is this->lens[mode[i]], should either all be vectors or be matices with same orientation
   * \param[in] aux_mode_first if true k-dim mode is first in all matrices in mat_list
   */
  template <typename dtype>
  void TTTP(Tensor<dtype> * T, int num_ops, int const * modes, Tensor<dtype> ** mat_list, bool aux_mode_first=false);

  /*
   * \brief calculates the singular value decomposition, M = U x S x VT, of matrix (unfolding of this tensor) using pdgesvd from ScaLAPACK
   * \param[in] idx_A char array of length order specifying tensor indices
   * \param[out] U left singular vectors of matrix
   * \param[in] idx_U char array of length num_U_modes+1 (with terminating character) specifying which of the tensor indices enumerate rows of unfolded tensor
   * \param[out] S singular values of matrix
   * \param[in] idx_V char array of length order-num_U_modes+1 (with terminating character) specifying which of the tensor indices enumerate columns of unfolded tensor
   * \param[out] VT right singular vectors of matrix
   * \param[in] rank rank of output matrices. If rank = 0, will use min(matrix.rows, matrix.columns)
   * \param[in] threshold for truncating singular values of the SVD, determines rank, if used, must set previous paramter rank=0
   * \param[in] use_svd_rand if true, use randomized SVD, in which case rank must be prespecified as opposed to threshold
   * \param[in] iter number of orthogonal iterations to perform (higher gives better accuracy) for randomized SVD
   * \param[in] oversamp oversampling parameter for randomized SVD
   */
  template <typename dtype>
  void svd(Tensor<dtype> & dA, char const * idx_A, Idx_Tensor & U, Idx_Tensor & S, Idx_Tensor & VT, int rank=0, double threshold=0., bool use_svd_rand=false, int num_iter=1, int oversamp=5);


}

#endif
