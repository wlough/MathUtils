template <typename DataType, MeshIndex Ncols> struct MeshDataStatic {
  static_assert(std::is_trivially_copyable_v<DataType>);
  static_assert(Ncols >= 1, "Ncols must be at least 1");
  MeshIndex id_index{
      MESH_ID_UNASSIGNED}; // ID of the thing this data is attached to
  MeshIndex id_data{MESH_ID_UNASSIGNED}; // ID of the thing this data represents
  std::vector<DataType> data; // row-major plays nice with numpy ravel/reshape

  MeshDataStatic() = default;
  ~MeshDataStatic() = default;
  MeshDataStatic(MeshIndex id_index_, MeshIndex id_data_)
      : id_index(id_index_), id_data(id_data_) {}
  MeshDataStatic(MeshID id_index_, MeshID id_data_,
                 const std::vector<DataType> &data_)
      : id_index(id_index_), id_data(id_data_), data(data_) {
    if (data.size() % Ncols != 0)
      throw std::invalid_argument("MeshDataStatic size mismatch");
  }

  /**
   * @brief Get number of rows
   *
   * @return MeshID
   */
  MeshID rows() const {
    assert(data.size() % Ncols == 0); // DEBUG
    return static_cast<MeshID>(data.size() / Ncols);
  }
  MeshID cols() const { return Ncols; }

  /**
   * @brief Access element (i, j)
   *
   * @param i
   * @param j
   * @return DataType& to element (i, j)
   */
  DataType &operator()(MeshID i, MeshID j) { return data[i * Ncols + j]; }
  /**
   * @brief Access element (i, j) (const version)
   *
   * @param i
   * @param j
   * @return const DataType& to element (i, j)
   */
  const DataType &operator()(MeshID i, MeshID j) const {
    return data[i * Ncols + j];
  }
  /**
   * @brief Access element i
   *
   * @param i
   * @return DataType& to element i
   */
  DataType &operator[](MeshID i) { return data[i]; }
  /**
   * @brief Access element i (const version)
   *
   * @param i
   * @return const DataType& to element i
   */
  const DataType &operator[](MeshID i) const { return data[i]; }

  /**
   * @brief Access element (i, j) with bounds checking
   *
   * @param i
   * @param j
   * @return DataType& to element (i, j)
   */
  DataType &at(MeshID i, MeshID j) {
    if (i >= rows() || j >= Ncols)
      throw std::out_of_range("MeshDataStatic::at");
    return operator()(i, j);
  }
  /**
   * @brief Access element (i, j) with bounds checking (const version)
   *
   * @param i
   * @param j
   * @return const DataType& to element (i, j)
   */
  const DataType &at(MeshID i, MeshID j) const {
    if (i >= rows() || j >= Ncols)
      throw std::out_of_range("MeshDataStatic::at");
    return operator()(i, j);
  }
  /**
   * @brief Resize the data to new_rows
   *
   * @param new_rows
   */
  void resize_rows(MeshID new_rows) {
    std::size_t new_size =
        static_cast<std::size_t>(new_rows) * static_cast<std::size_t>(Ncols);
    data.resize(new_size);
  }
  /**
   * @brief Clear data
   *
   */
  void clear() { data.clear(); }
  /**
   * @brief Reserve space for new_rows
   *
   * @param new_rows
   */
  void reserve_rows(MeshID new_rows) {
    std::size_t new_size =
        static_cast<std::size_t>(new_rows) * static_cast<std::size_t>(Ncols);
    data.reserve(new_size);
  }
  /**
   * @brief Shrink data to fit
   *
   */
  void shrink_to_fit() { data.shrink_to_fit(); }

  /**
   * @brief Get raw data pointer
   *
   * @return DataType*
   */
  DataType *data_ptr() { return data.data(); }
  /**
   * @brief Get raw data pointer (const version)
   *
   * @return const DataType*
   */
  const DataType *data_ptr() const { return data.data(); }
  /**
   * @brief Get total size (number of elements)
   *
   * @return MeshID
   */
  MeshID size() const {
    assert(data.size() <= std::numeric_limits<MeshID>::max()); // DEBUG
    return static_cast<MeshID>(data.size());
  }
  auto begin() { return data.begin(); }
  auto end() { return data.end(); }
  auto begin() const { return data.begin(); }
  auto end() const { return data.end(); }

  /**
   * @brief Extract column j as a MeshDataStatic<DataType, 1>
   *
   * @param j
   * @return MeshDataStatic<DataType, 1>
   */
  MeshDataStatic<DataType, 1> col(MeshID j) const {
    if (j >= Ncols)
      throw std::out_of_range("MeshDataStatic::col");
    MeshDataStatic<DataType, 1> col_data(id_index, id_data);
    col_data.data.resize(rows());
    for (MeshID i = 0; i < rows(); ++i) {
      col_data.data[i] = operator()(i, j);
    }
    return col_data;
  }

  /**
   * @brief Get pointer to the start of row i
   *
   * @param i
   * @return DataType* to row i
   */
  DataType *row_ptr(MeshID i) {
    if (i >= rows())
      throw std::out_of_range("MeshDataStatic::row_ptr");
    return data.data() + static_cast<std::size_t>(i) * Ncols;
  }
  /**
   * @brief Get pointer to the start of row i (const version)
   *
   * @param i
   * @return const DataType* to row i
   */
  const DataType *row_ptr(MeshID i) const {
    if (i >= rows())
      throw std::out_of_range("MeshDataStatic::row_ptr");
    return data.data() + static_cast<std::size_t>(i) * Ncols;
  }

  /**
   * @brief Attempt to convert to MeshDataStatic<NewDataType, Ncols>
   */
  template <typename NewDataType>
  MeshDataStatic<NewDataType, Ncols> to_dtype() const {
    static_assert(std::is_convertible<DataType, NewDataType>::value,
                  "DataType must be convertible to NewDataType");
    MeshDataStatic<NewDataType, Ncols> new_data(id_index, id_data);
    new_data.data.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      new_data.data[i] = static_cast<NewDataType>(data[i]);
    }
    return new_data;
  }

  // ---- Eigen types ----
  using EigMatFixed =
      Eigen::Matrix<DataType, Eigen::Dynamic, Ncols, Eigen::RowMajor>;
  using EigMapFixedConst = Eigen::Map<const EigMatFixed>;
  using EigMapFixed = Eigen::Map<EigMatFixed>;
  /**
   * @brief Copy into an owning Eigen matrix
   */
  EigMatFixed to_eigen() const {
    EigMatFixed mat(static_cast<Eigen::Index>(rows()),
                    static_cast<Eigen::Index>(Ncols));
    std::memcpy(mat.data(), data.data(), data.size() * sizeof(DataType));
    return mat;
  }
  /**
   * @brief Return a zero-copy Eigen::Map view of the data (mutable)
   */
  EigMapFixed as_eigen() {
    return {data.data(), (Eigen::Index)rows(), (Eigen::Index)Ncols};
  }
  /**
   * @brief Return a zero-copy Eigen::Map view of the data (const)
   */
  EigMapFixedConst as_eigen() const {
    return {data.data(), (Eigen::Index)rows(), (Eigen::Index)Ncols};
  }
};

template <typename DataType> struct MeshData {

  static_assert(std::is_trivially_copyable_v<DataType>);

  MeshID index_id_type{
      MESH_ID_UNASSIGNED}; // ID of the thing this data is attached to
  MeshID data_id_type{
      MESH_ID_UNASSIGNED}; // ID of the thing this data represents
  MeshID Ncols{1};
  std::vector<DataType> data; // row-major plays nice with numpy ravel/reshape

  MeshData() = default;
  ~MeshData() = default;
  MeshData(MeshID ncols_) : Ncols(ncols_) {
    if (Ncols < 1)
      throw std::invalid_argument("MeshData Ncols < 1");
  }
  MeshData(MeshID index_id_type_, MeshID data_id_type_, MeshID ncols_)
      : index_id_type(index_id_type_), data_id_type(data_id_type_),
        Ncols(ncols_) {
    if (Ncols < 1)
      throw std::invalid_argument("MeshData Ncols < 1");
  }
  MeshData(MeshID ncols_, const std::vector<DataType> &data_)
      : Ncols(ncols_), data(data_) {
    if (Ncols < 1 || data.size() % Ncols != 0)
      throw std::invalid_argument("MeshData Ncols < 1 or size mismatch");
  }

  /**
   * @brief Get number of rows
   *
   * @return MeshID
   */
  MeshID rows() const {
    assert(data.size() % Ncols == 0);                                  // DEBUG
    assert(data.size() / Ncols <= std::numeric_limits<MeshID>::max()); // DEBUG
    return static_cast<MeshID>(data.size() / Ncols);
  }
  MeshID cols() const { return Ncols; }

  /**
   * @brief Access element (i, j)
   *
   * @param i
   * @param j
   * @return DataType& to element (i, j)
   */
  DataType &operator()(MeshID i, MeshID j) { return data[i * Ncols + j]; }
  /**
   * @brief Access element (i, j) (const version)
   *
   * @param i
   * @param j
   * @return const DataType& to element (i, j)
   */
  const DataType &operator()(MeshID i, MeshID j) const {
    return data[i * Ncols + j];
  }
  /**
   * @brief Access element i
   *
   * @param i
   * @return DataType& to element i
   */
  DataType &operator[](MeshID i) { return data[i]; }
  /**
   * @brief Access element i (const version)
   *
   * @param i
   * @return const DataType& to element i
   */
  const DataType &operator[](MeshID i) const { return data[i]; }
  /**
   * @brief Access element (i, j) with bounds checking
   *
   * @param i
   * @param j
   * @return DataType& to element (i, j)
   */
  DataType &at(MeshID i, MeshID j) {
    if (i >= rows() || j >= Ncols)
      throw std::out_of_range("MeshData::at");
    return operator()(i, j);
  }
  /**
   * @brief Access element (i, j) with bounds checking (const version)
   *
   * @param i
   * @param j
   * @return const DataType& to element (i, j)
   */
  const DataType &at(MeshID i, MeshID j) const {
    if (i >= rows() || j >= Ncols)
      throw std::out_of_range("MeshData::at");
    return operator()(i, j);
  }
  /**
   * @brief Resize the data to new_rows
   *
   * @param new_rows
   */
  void resize_rows(MeshID new_rows) {
    std::size_t new_size =
        static_cast<std::size_t>(new_rows) * static_cast<std::size_t>(Ncols);
    data.resize(new_size);
  }
  /**
   * @brief Clear data
   *
   */
  void clear() { data.clear(); }
  /**
   * @brief Reserve space for new_rows
   *
   * @param new_rows
   */
  void reserve_rows(MeshID new_rows) {
    std::size_t new_size =
        static_cast<std::size_t>(new_rows) * static_cast<std::size_t>(Ncols);
    data.reserve(new_size);
  }
  /**
   * @brief Shrink data to fit
   *
   */
  void shrink_to_fit() { data.shrink_to_fit(); }

  /**
   * @brief Get raw data pointer
   *
   * @return DataType*
   */
  DataType *data_ptr() { return data.data(); }
  /**
   * @brief Get raw data pointer (const version)
   *
   * @return const DataType*
   */
  const DataType *data_ptr() const { return data.data(); }
  /**
   * @brief Get total size (number of elements)
   *
   * @return MeshID
   */
  MeshID size() const {
    assert(data.size() <= std::numeric_limits<MeshID>::max()); // DEBUG
    return static_cast<MeshID>(data.size());
  }
  auto begin() { return data.begin(); }
  auto end() { return data.end(); }
  auto begin() const { return data.begin(); }
  auto end() const { return data.end(); }

  /**
   * @brief Extract column j as a MeshData<DataType>
   *
   * @param j
   * @return MeshData<DataType>
   */
  MeshData<DataType> col(MeshID j) const {
    if (j >= Ncols)
      throw std::out_of_range("MeshData::col");
    MeshData<DataType> col_data(id_index, id_data, 1);
    col_data.data.resize(rows());
    for (MeshID i = 0; i < rows(); ++i) {
      col_data.data[i] = operator()(i, j);
    }
    return col_data;
  }
  /**
   * @brief Get pointer to the start of row i
   *
   * @param i
   * @return DataType*
   */
  DataType *row_ptr(MeshID i) {
    if (i >= rows())
      throw std::out_of_range("MeshData::row_ptr");
    return data.data() + static_cast<std::size_t>(i) * Ncols;
  }
  /**
   * @brief Get pointer to the start of row i (const version)
   *
   * @param i
   * @return const DataType*
   */
  const DataType *row_ptr(MeshID i) const {
    if (i >= rows())
      throw std::out_of_range("MeshData::row_ptr");
    return data.data() + static_cast<std::size_t>(i) * Ncols;
  }
  /**
   * @brief Attempt to convert to MeshData<NewDataType>
   */
  template <typename NewDataType> MeshData<NewDataType> to_dtype() const {
    static_assert(std::is_convertible<DataType, NewDataType>::value,
                  "DataType must be convertible to NewDataType");
    MeshData<NewDataType> new_data(id_index, id_data, Ncols);
    new_data.data.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      new_data.data[i] = static_cast<NewDataType>(data[i]);
    }
    return new_data;
  }

  // ---- Eigen types ----
  using EigMat =
      Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using EigMap = Eigen::Map<EigMat>;
  using EigMapConst = Eigen::Map<const EigMat>;
  template <MeshID C>
  using EigMatFixed =
      Eigen::Matrix<DataType, Eigen::Dynamic, C, Eigen::RowMajor>;
  /**
   * @brief Copy into an owning Eigen matrix
   */
  EigMat to_eigen() const {
    EigMat mat(static_cast<Eigen::Index>(rows()),
               static_cast<Eigen::Index>(Ncols));
    std::memcpy(mat.data(), data.data(), data.size() * sizeof(DataType));
    return mat;
  }
  /**
   * @brief Return a zero-copy Eigen::Map view of the data (mutable)
   */
  EigMap as_eigen() {
    return EigMap(data.data(), static_cast<Eigen::Index>(rows()),
                  static_cast<Eigen::Index>(Ncols));
  }
  /**
   * @brief Return a zero-copy Eigen::Map view of the data (const)
   */
  EigMapConst as_eigen() const {
    return {data.data(), (Eigen::Index)rows(), (Eigen::Index)Ncols};
  }
  /**
   * @brief Return a zero-copy Eigen::Map view of the data with fixed columns
   * (mutable)
   */
  template <MeshID C> Eigen::Map<EigMatFixed<C>> as_eigen_fixed() {
    if (Ncols != C)
      throw std::runtime_error("MeshData::as_eigen_fixed: Ncols mismatch");
    return {data.data(), (Eigen::Index)rows(), (Eigen::Index)C};
  }
  /**
   * @brief Return a zero-copy Eigen::Map view of the data with fixed columns
   * (const)
   */
  template <MeshID C> Eigen::Map<const EigMatFixed<C>> as_eigen_fixed() const {
    if (Ncols != C)
      throw std::runtime_error("MeshData::as_eigen_fixed: Ncols mismatch");
    return {data.data(), (Eigen::Index)rows(), (Eigen::Index)C};
  }
};
