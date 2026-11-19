
  vec3 rideFetch(sampler2D t, int i) { return texelFetch(t, ivec2(i & 4095, i >> 12), 0).xyz; }
  mat3 rideFrame(vec3 a, vec3 b, vec3 c) {
    vec3 e1 = normalize(b - a);
    vec3 n = normalize(cross(b - a, c - a));
    return mat3(e1, cross(n, e1), n);
  }
  vec4 rideQuat(mat3 m) {  // m[col][row]
    float t = m[0][0] + m[1][1] + m[2][2];
    if (t > 0.0) { float s = sqrt(t + 1.0) * 2.0;
      return vec4((m[1][2] - m[2][1]) / s, (m[2][0] - m[0][2]) / s, (m[0][1] - m[1][0]) / s, 0.25 * s); }
    if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) { float s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
      return vec4(0.25 * s, (m[1][0] + m[0][1]) / s, (m[2][0] + m[0][2]) / s, (m[1][2] - m[2][1]) / s); }
    if (m[1][1] > m[2][2]) { float s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
      return vec4((m[1][0] + m[0][1]) / s, 0.25 * s, (m[2][1] + m[1][2]) / s, (m[2][0] - m[0][2]) / s); }
    float s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
    return vec4((m[2][0] + m[0][2]) / s, (m[2][1] + m[1][2]) / s, 0.25 * s, (m[0][1] - m[1][0]) / s);
  }
  vec4 rideMul(vec4 a, vec4 b) { return vec4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz)); }
  void rideTriangle(sampler2D bind, sampler2D tris, sampler2D rest, sampler2D cur, int index,
                    inout vec3 center, inout vec4 quaternion) {
    vec4 bd = texelFetch(bind, ivec2(index & 4095, index >> 12), 0);
    int t = int(bd.x + 0.5);
    vec3 ids = texelFetch(tris, ivec2(t & 4095, t >> 12), 0).xyz;
    int i0 = int(ids.x + 0.5), i1 = int(ids.y + 0.5), i2 = int(ids.z + 0.5);
    vec3 a0 = rideFetch(rest, i0), b0 = rideFetch(rest, i1), c0 = rideFetch(rest, i2);
    vec3 a1 = rideFetch(cur, i0), b1 = rideFetch(cur, i1), c1 = rideFetch(cur, i2);
    mat3 r0 = rideFrame(a0, b0, c0), r1 = rideFrame(a1, b1, c1);
    vec3 p0 = a0 + bd.y * (b0 - a0) + bd.z * (c0 - a0);
    vec3 p1 = a1 + bd.y * (b1 - a1) + bd.z * (c1 - a1);
    center = p1 + r1 * (transpose(r0) * (center - p0));
    quaternion = normalize(rideMul(rideQuat(r1 * transpose(r0)), quaternion));
  }
