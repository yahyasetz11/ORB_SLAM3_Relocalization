#pragma once

// osa_to_csv is a headless tool — it never opens a window or calls any GL
// function at runtime. However, ORB-SLAM3 headers pull in Pangolin headers
// (MapDrawer.h → <pangolin/pangolin.h>, Map.h → <pangolin/pangolin.h>), and
// Pangolin's glplatform.h selects its GL backend via compile-time defines.
//
// Without HAVE_EPOXY, glplatform.h falls back to bare <GL/gl.h> (OpenGL 1.1
// only), which is missing glBindAttribLocation (GL 2.0), glGetProgramResourceIndex
// and glShaderStorageBlockBinding (GL 4.3) — the symbols referenced in
// pangolin/gl/glsl.hpp.
//
// Pangolin on this system was built against libepoxy (libpango_opengl.so.0
// links libepoxy.so.0). Defining HAVE_EPOXY here makes glplatform.h take the
// epoxy path, which declares all modern GL functions as real prototypes.
// Epoxy also defines __gl_h_, preventing the subsequent bare <GL/gl.h>
// include in glplatform.h from re-entering.
//
// This header must be the first include in osa_to_csv.cpp.

#define HAVE_EPOXY
#include <epoxy/gl.h>
