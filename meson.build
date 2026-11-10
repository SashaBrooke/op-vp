project('opvp', 'cpp',
  version: '1.0.0',
  default_options: ['cpp_std=c++17', 'buildtype=release', 'werror=false', 'warning_level=3']
)

threads_dep = dependency('threads')
dl_dep = dependency('dl', required: true)
boost_dep = dependency('boost', modules: ['program_options'], required: true)
opencv_dep = dependency('opencv4', required: true)
ort_dep = declare_dependency(link_args: ['-lonnxruntime'])

compile_deps = [threads_dep, dl_dep, boost_dep, opencv_dep, ort_dep]

project_inc_dirs = include_directories(
  'src',
  'src/engine',
  'src/video_source'
)

target_platform = get_option('platform')

if target_platform == 'rpi5'
  add_project_arguments('-DPLATFORM_RPI5', language: 'cpp')
  libcamera_dep = dependency('libcamera', required: false)
  if not libcamera_dep.found()
    libcamera_dep = declare_dependency(link_args: ['-lcamera', '-lcamera-base'])
  endif
  rpicam_dep = declare_dependency(link_args: ['-lrpicam_app'])
  compile_deps += [libcamera_dep, rpicam_dep]
  project_inc_dirs += include_directories('/usr/local/include/rpicam-apps')
else
  add_project_arguments('-DPLATFORM_DESKTOP', language: 'cpp')
endif

src_files = files(
  'src/main.cpp',
  'src/engine/cpu_engine.cpp',
  'src/engine/gpu_engine.cpp',
  'src/engine/hailo_engine.cpp',
  'src/engine/engine_factory.cpp',
  'src/video_source/file_source.cpp',
  'src/video_source/camera_source.cpp',
  'src/video_source/video_source_factory.cpp'
)

exe = executable('opvp',
  src_files,
  include_directories: project_inc_dirs,
  dependencies: compile_deps,
  install: true,
  install_dir: '/opt/opvp'
)

install_subdir('models', install_dir: '/opt/opvp')
install_subdir('media', install_dir: '/opt/opvp')
install_emptydir('/opt/opvp/output', install_mode: 'rwxrwxrwx')
