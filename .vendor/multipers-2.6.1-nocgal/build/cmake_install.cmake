# Install script for directory: /gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/tmp/tmps1e6q4qp/wheel/platlib")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_slicer_nanobind.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_slicer_nanobind.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_slicer_nanobind.cpython-311-x86_64-linux-gnu.so"
         RPATH [[$ORIGIN]])
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/multipers" TYPE MODULE FILES "/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers/_slicer_nanobind.cpython-311-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_slicer_nanobind.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_slicer_nanobind.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_slicer_nanobind.cpython-311-x86_64-linux-gnu.so"
         OLD_RPATH [[$ORIGIN:/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers:/share/apps/rocky9/spack/apps/linux-rocky9-x86_64_v4/gcc-12.2.0/intel-tbb/2021.9.0-7fsrvwa/lib64:]]
         NEW_RPATH [[$ORIGIN]])
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_slicer_nanobind.cpython-311-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_mma_nanobind.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_mma_nanobind.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_mma_nanobind.cpython-311-x86_64-linux-gnu.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/multipers" TYPE MODULE FILES "/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers/_mma_nanobind.cpython-311-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_mma_nanobind.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_mma_nanobind.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_mma_nanobind.cpython-311-x86_64-linux-gnu.so"
         OLD_RPATH "/share/apps/rocky9/spack/apps/linux-rocky9-x86_64_v4/gcc-12.2.0/intel-tbb/2021.9.0-7fsrvwa/lib64:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_mma_nanobind.cpython-311-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_simplex_tree_multi_nanobind.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_simplex_tree_multi_nanobind.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_simplex_tree_multi_nanobind.cpython-311-x86_64-linux-gnu.so"
         RPATH [[$ORIGIN]])
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/multipers" TYPE MODULE FILES "/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers/_simplex_tree_multi_nanobind.cpython-311-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_simplex_tree_multi_nanobind.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_simplex_tree_multi_nanobind.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_simplex_tree_multi_nanobind.cpython-311-x86_64-linux-gnu.so"
         OLD_RPATH [[$ORIGIN:/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers:/share/apps/rocky9/spack/apps/linux-rocky9-x86_64_v4/gcc-12.2.0/intel-tbb/2021.9.0-7fsrvwa/lib64:]]
         NEW_RPATH [[$ORIGIN]])
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_simplex_tree_multi_nanobind.cpython-311-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_function_rips_nanobind.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_function_rips_nanobind.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_function_rips_nanobind.cpython-311-x86_64-linux-gnu.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/multipers" TYPE MODULE FILES "/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers/_function_rips_nanobind.cpython-311-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_function_rips_nanobind.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_function_rips_nanobind.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_function_rips_nanobind.cpython-311-x86_64-linux-gnu.so"
         OLD_RPATH "/share/apps/rocky9/spack/apps/linux-rocky9-x86_64_v4/gcc-12.2.0/intel-tbb/2021.9.0-7fsrvwa/lib64:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_function_rips_nanobind.cpython-311-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_grid_helper_nanobind.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_grid_helper_nanobind.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_grid_helper_nanobind.cpython-311-x86_64-linux-gnu.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/multipers" TYPE MODULE FILES "/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers/_grid_helper_nanobind.cpython-311-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_grid_helper_nanobind.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_grid_helper_nanobind.cpython-311-x86_64-linux-gnu.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_grid_helper_nanobind.cpython-311-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_mpfree_interface.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_mpfree_interface.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_mpfree_interface.cpython-311-x86_64-linux-gnu.so"
         RPATH [[$ORIGIN]])
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/multipers" TYPE MODULE FILES "/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers/_mpfree_interface.cpython-311-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_mpfree_interface.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_mpfree_interface.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_mpfree_interface.cpython-311-x86_64-linux-gnu.so"
         OLD_RPATH [[$ORIGIN:/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers:/share/apps/rocky9/spack/apps/linux-rocky9-x86_64_v4/gcc-12.2.0/boost/1.85.0-7yb3mhc/lib:/share/apps/rocky9/spack/apps/linux-rocky9-x86_64_v4/gcc-12.2.0/intel-tbb/2021.9.0-7fsrvwa/lib64:/gpfs/scratch/qp252676/globus/envs/cloud-env/lib:]]
         NEW_RPATH [[$ORIGIN]])
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_mpfree_interface.cpython-311-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_function_delaunay_interface.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_function_delaunay_interface.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_function_delaunay_interface.cpython-311-x86_64-linux-gnu.so"
         RPATH [[$ORIGIN]])
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/multipers" TYPE MODULE FILES "/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers/_function_delaunay_interface.cpython-311-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_function_delaunay_interface.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_function_delaunay_interface.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_function_delaunay_interface.cpython-311-x86_64-linux-gnu.so"
         OLD_RPATH [[$ORIGIN:/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers:/share/apps/rocky9/spack/apps/linux-rocky9-x86_64_v4/gcc-12.2.0/boost/1.85.0-7yb3mhc/lib:/share/apps/rocky9/spack/apps/linux-rocky9-x86_64_v4/gcc-12.2.0/intel-tbb/2021.9.0-7fsrvwa/lib64:/gpfs/scratch/qp252676/globus/envs/cloud-env/lib:]]
         NEW_RPATH [[$ORIGIN]])
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_function_delaunay_interface.cpython-311-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_2pac_interface.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_2pac_interface.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_2pac_interface.cpython-311-x86_64-linux-gnu.so"
         RPATH [[$ORIGIN]])
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/multipers" TYPE MODULE FILES "/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers/_2pac_interface.cpython-311-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_2pac_interface.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_2pac_interface.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_2pac_interface.cpython-311-x86_64-linux-gnu.so"
         OLD_RPATH [[$ORIGIN:/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers:/share/apps/rocky9/spack/apps/linux-rocky9-x86_64_v4/gcc-12.2.0/intel-tbb/2021.9.0-7fsrvwa/lib64:]]
         NEW_RPATH [[$ORIGIN]])
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_2pac_interface.cpython-311-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_hera_interface.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_hera_interface.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_hera_interface.cpython-311-x86_64-linux-gnu.so"
         RPATH [[$ORIGIN]])
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/multipers" TYPE MODULE FILES "/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers/_hera_interface.cpython-311-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_hera_interface.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_hera_interface.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_hera_interface.cpython-311-x86_64-linux-gnu.so"
         OLD_RPATH [[$ORIGIN:/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers:/share/apps/rocky9/spack/apps/linux-rocky9-x86_64_v4/gcc-12.2.0/intel-tbb/2021.9.0-7fsrvwa/lib64:]]
         NEW_RPATH [[$ORIGIN]])
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_hera_interface.cpython-311-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_multi_critical_interface.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_multi_critical_interface.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_multi_critical_interface.cpython-311-x86_64-linux-gnu.so"
         RPATH [[$ORIGIN]])
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/multipers" TYPE MODULE FILES "/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers/_multi_critical_interface.cpython-311-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_multi_critical_interface.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_multi_critical_interface.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_multi_critical_interface.cpython-311-x86_64-linux-gnu.so"
         OLD_RPATH [[$ORIGIN:/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers:/share/apps/rocky9/spack/apps/linux-rocky9-x86_64_v4/gcc-12.2.0/boost/1.85.0-7yb3mhc/lib:/share/apps/rocky9/spack/apps/linux-rocky9-x86_64_v4/gcc-12.2.0/intel-tbb/2021.9.0-7fsrvwa/lib64:/gpfs/scratch/qp252676/globus/envs/cloud-env/lib:]]
         NEW_RPATH [[$ORIGIN]])
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_multi_critical_interface.cpython-311-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_rhomboid_tiling_interface.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_rhomboid_tiling_interface.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_rhomboid_tiling_interface.cpython-311-x86_64-linux-gnu.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/multipers" TYPE MODULE FILES "/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers/_rhomboid_tiling_interface.cpython-311-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_rhomboid_tiling_interface.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_rhomboid_tiling_interface.cpython-311-x86_64-linux-gnu.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_rhomboid_tiling_interface.cpython-311-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_aida_interface.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_aida_interface.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_aida_interface.cpython-311-x86_64-linux-gnu.so"
         RPATH [[$ORIGIN]])
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/multipers" TYPE MODULE FILES "/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers/_aida_interface.cpython-311-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_aida_interface.cpython-311-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_aida_interface.cpython-311-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_aida_interface.cpython-311-x86_64-linux-gnu.so"
         OLD_RPATH [[$ORIGIN:/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers:/share/apps/rocky9/spack/apps/linux-rocky9-x86_64_v4/gcc-12.2.0/boost/1.85.0-7yb3mhc/lib:/share/apps/rocky9/spack/apps/linux-rocky9-x86_64_v4/gcc-12.2.0/intel-tbb/2021.9.0-7fsrvwa/lib64:/gpfs/scratch/qp252676/globus/envs/cloud-env/lib:]]
         NEW_RPATH [[$ORIGIN]])
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/_aida_interface.cpython-311-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/libmultipers_core.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/libmultipers_core.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/libmultipers_core.so"
         RPATH [[$ORIGIN]])
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/multipers" TYPE SHARED_LIBRARY FILES "/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/compiled_modules/multipers/libmultipers_core.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/libmultipers_core.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/libmultipers_core.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/libmultipers_core.so"
         OLD_RPATH [[$ORIGIN:/share/apps/rocky9/spack/apps/linux-rocky9-x86_64_v4/gcc-12.2.0/intel-tbb/2021.9.0-7fsrvwa/lib64:]]
         NEW_RPATH [[$ORIGIN]])
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/multipers/libmultipers_core.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/multipers" TYPE DIRECTORY FILES "/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/multipers/" REGEX "/\\.DS\\_Store$" EXCLUDE REGEX "/\\_\\_pycache\\_\\_$" EXCLUDE REGEX "/[^/]*\\.pyc$" EXCLUDE REGEX "/[^/]*\\.pyo$" EXCLUDE REGEX "/[^/]*\\.dep$" EXCLUDE REGEX "/[^/]*\\.pyx$" EXCLUDE REGEX "/[^/]*\\.pxd$" EXCLUDE REGEX "/[^/]*\\.tp$" EXCLUDE REGEX "/[^/]*\\.cpp$" EXCLUDE REGEX "/[^/]*\\.h$" EXCLUDE REGEX "/[^/]*\\.hpp$" EXCLUDE)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/gpfs/scratch/qp252676/globus/point-process-tda/.vendor/multipers-2.6.1-nocgal/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
