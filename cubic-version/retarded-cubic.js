/*

A single charged particle follows a general path defined by vertices (x, y) in a float buffer.
The position and velocity and acceleration at any time t is obtained by cubic interpolation in the buffer.
The buffer is periodic. The interpolated path will be continuous up to the first derivative (velocity).

*/

function catmull_rom_coefs(f) {
  const fn1 = f[0];
  const f0 = f[1];
  const f1 = f[2];
  const f2 = f[3];
  const c0 = 2.0 * f0;
  const c1 = -1.0 * fn1 + 1.0 * f1;
  const c2 = 2.0 * fn1 - 5.0 * f0 + 4.0 * f1 - 1.0 * f2;
  const c3 = -1.0 * fn1 + 3.0 * f0 - 3.0 * f1 + 1.0 * f2;
  return [c0, c1, c2, c3];
}

function spline_value(c, t) {
  const t2 = t * t;
  return 0.5 * (c[0] + c[1] * t + c[2] * t2 + c[3] * t2 * t);
}

function spline_derivative(c, t) {
  const t2 = t * t;
  return 0.5 * (c[1] + 2.0 * c[2] * t + 3.0 * c[3] * t2);
}

function cubic_interp(theta, omega, vertices) {
  const npts = vertices.length / 2;
  const delta_theta = 2.0 * Math.PI / npts;
  const dscale = omega / delta_theta;
  while (theta < -Math.PI) theta += 2 * Math.PI;
  while (theta > Math.PI) theta -= 2 * Math.PI;
  const z = (theta + Math.PI) / delta_theta;
  const idx = Math.floor(z);
  const w = z - idx;
  const in1 = (idx + npts - 1) % npts;
  const i0 = idx % npts;
  const i1 = (idx + 1) % npts;
  const i2 = (idx + 2) % npts;
  const cx = catmull_rom_coefs([vertices[2 * in1 + 0], vertices[2 * i0 + 0], vertices[2 * i1 + 0], vertices[2 * i2 + 0]]);
  const cy = catmull_rom_coefs([vertices[2 * in1 + 1], vertices[2 * i0 + 1], vertices[2 * i1 + 1], vertices[2 * i2 + 1]]);
  const rx = spline_value(cx, w);
  const ry = spline_value(cy, w);
  const vx = spline_derivative(cx, w) * dscale;
  const vy = spline_derivative(cy, w) * dscale;
  return [rx, ry, vx, vy];
}

function find_maximum_velocity(omega, vertices) {
  const npts = vertices.length / 2;
  const nsamples = 5 * npts; // oversample the path
  let max_vsq = 0.0;
  for (let i = 0; i < nsamples; i++) {
    const thetai = i * 2 * Math.PI / nsamples;
    const rv = cubic_interp(thetai, omega, vertices);
    const vsq = rv[2] * rv[2] + rv[3] * rv[3];
    if (vsq > max_vsq) max_vsq = vsq;
  }
  return Math.sqrt(max_vsq);
}

function canvas_coord(xy, wh, z) {
  const aspect = wh[0] / wh[1];
  const hw = wh[0] / 2;
  const hh = wh[1] / 2;
  const x = hw + xy[0] * hw / aspect / z;
  const y = hh - xy[1] * hh / z;
  return [x, y];
}

const numSourceVertices = 32; // need to match the number in the shader code
const sourceVertices = new Float32Array(2 * numSourceVertices);

function preset_path_0(theta) {
  return [Math.cos(theta), Math.sin(theta)];
}

function preset_path_1(theta) {
  const shift = 0.01;
  return [Math.sign(Math.cos(theta + shift)), Math.sign(Math.sin(theta + shift))];
}

function preset_path_2(theta) {
  const xi = theta / (2 * Math.PI);
  return [xi, xi];
}

function preset_path_3(theta) {
  return [Math.cos(2.0 * theta), Math.sin(3.0 * theta)];
}

function preset_path_4(theta) {
  const r = 2.0 + 0.25 * Math.cos(10 * theta);
  return [r * Math.cos(theta), r * Math.sin(theta)];
}

function preset_path_5(theta) {
  const r = 2.0;
  const phase = 0.10 * Math.cos(10 * theta)
  return [r * Math.cos(theta + phase), r * Math.sin(theta + phase)];
}

function preset_path_6(theta) {
  const cs = Math.cos(theta);
  const sn = Math.sin(theta);
  return [cs * Math.sqrt(Math.abs(cs)), sn * Math.sqrt(Math.abs(sn))];
}

function preset_path_7(theta) {
  return [3.0 * Math.sin(theta), 0.0];
}

const PATH_FUNCTIONS = [
  preset_path_0,
  preset_path_1,
  preset_path_2,
  preset_path_3,
  preset_path_4,
  preset_path_5,
  preset_path_6,
  preset_path_7
];

function generate_preset_path(idx) {
  const delta_theta = 2.0 * Math.PI / numSourceVertices;
  const fxy = PATH_FUNCTIONS[idx];
  for (let i = 0; i < numSourceVertices; i++) {
    k = 2 * i;
    let thetai = -Math.PI + i * delta_theta;
    let xy = fxy(thetai);
    sourceVertices[k + 0] = xy[0];
    sourceVertices[k + 1] = xy[1];
  }
}

let IPATH = 1;
generate_preset_path(IPATH);
let VMAX = find_maximum_velocity(1.0, sourceVertices);
console.log(VMAX);

// const rrvv = cubic_interp(0.0, 1.0, sourceVertices);
// console.log(rrvv); // should show [1, 0, 0, 1]

// Get canvas and context
const canvasgl = document.getElementById('gl-canvas');
const gl = canvasgl.getContext('webgl2');
if (!gl) {
  alert('WebGL2 not supported');
  throw new Error('WebGL2 not supported');
}

// The standard "canvas" is for text overlay
const canvas2d = document.getElementById('2d-canvas');

// Set canvas to fullscreen
function resizeCanvas() {
  canvasgl.width = window.innerWidth;
  canvasgl.height = window.innerHeight;
  gl.viewport(0, 0, canvasgl.width, canvasgl.height);

  canvas2d.width = window.innerWidth;
  canvas2d.height = window.innerHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Vertex Shader
const vertexShaderSource = `#version 300 es
  in vec2 a_position;
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
  }
`;

// Fragment Shader
const fragmentShaderSource = `#version 300 es
  precision highp float;
  out vec4 fragColor;
  uniform vec2 u_resolution;
  uniform vec2 u_rho;
  uniform float u_time;
  uniform float u_zoom;
  uniform float u_contrast;
  uniform float u_vmax;
  uniform float u_beta;
  uniform float u_freq;
  uniform int u_style;

  // The number of spline points must be a power of two (for bitmask modulo op below)
  #define NUM_SOURCE_VERTICES 32
  uniform vec2 u_source_vertices[NUM_SOURCE_VERTICES];

  const float PI = 3.141592653589793;
  const float TWOPI = 2.0 * PI;
  const float delta_theta = TWOPI / float(NUM_SOURCE_VERTICES);

  vec2[4] catmull_rom_coefs(vec2 fn1, vec2 f0, vec2 f1, vec2 f2)
  {
    vec2[4] c;
    c[0] = 2.0 * f0;
    c[1] = -1.0 * fn1 + 1.0 * f1;
    c[2] = 2.0 * fn1 - 5.0 * f0 + 4.0 * f1 - 1.0 * f2;
    c[3] = -1.0 * fn1 + 3.0 * f0 - 3.0 * f1 + 1.0 * f2;
    return c;
  }

  vec2 spline_value(vec2[4] c, float t)
  {
    float t2 = t * t;
    return 0.5 * (c[0] + c[1] * t + c[2] * t2 + c[3] * t2 * t);
  }

  vec2 spline_derivative(vec2[4] c, float t)
  {
    float t2 = t * t;
    return 0.5 * (c[1] + 2.0 * c[2] * t + 3.0 * c[3] * t2);
  }

  vec2 spline_acceleration(vec2[4] c, float t)
  {
    return 0.5 * (2.0 * c[2] + 6.0 * c[3] * t);
  }

  vec2[3] cubic_interp(vec2 osc, float omega, bool only_position)
  {
    float theta = atan(osc.y, osc.x); // (-PI, PI]
    float z = (theta + PI) / delta_theta; // float index into array
    float w = fract(z);

    int i0 = int(z);
    vec2 fn1 = u_source_vertices[(i0 + NUM_SOURCE_VERTICES - 1) & (NUM_SOURCE_VERTICES - 1)];
    vec2 f0 = u_source_vertices[i0 & (NUM_SOURCE_VERTICES - 1)];
    vec2 f1 = u_source_vertices[(i0 + 1) & (NUM_SOURCE_VERTICES - 1)];
    vec2 f2 = u_source_vertices[(i0 + 2) & (NUM_SOURCE_VERTICES - 1)];

    vec2[4] c = catmull_rom_coefs(fn1, f0, f1, f2);

    vec2[3] rva;
    rva[0] = spline_value(c, w);
    if (only_position) return rva;
    float dscale = (omega / delta_theta);
    rva[1] = spline_derivative(c, w) * dscale;
    rva[2] = spline_acceleration(c, w) * dscale * dscale;
    return rva;
  }

  vec2 backroll_osc(vec2 osc, float omega, float tau) {
    float theta = -omega * tau;
    float cs = cos(theta);
    float sn = sin(theta);
    return vec2(cs * osc.x - sn * osc.y, sn * osc.x + cs * osc.y);
  }

  float root_function(vec2 osc, float omega, float c, vec2 pos, float tau) {
    vec2 osc_ = backroll_osc(osc, omega, tau);
    vec2[3] src = cubic_interp(osc_, omega, true);
    float dx = pos.x - src[0].x;
    float dy = pos.y - src[0].y;
    return sqrt(dx * dx + dy * dy) - c * tau;
  }

  // Bisect the retarded time tau for the present oscillator state and the present location 
  float bisect_tau(vec2 osc, float omega, float c, vec2 pos) {
    float tau0 = 0.0;
    float f0 = root_function(osc, omega, c, pos, tau0);
    if (f0 == 0.0) return tau0;
    float tau1 = f0 / c;
    float f1 = root_function(osc, omega, c, pos, tau1);
    while (f1 > 0.0) {
      tau0 = tau1;
      f0 = f1;
      tau1 *= 2.0;
      f1 = root_function(osc, omega, c, pos, tau1);
    };
    if (f1 == 0.0) return tau1;
    // Now f0 > 0 and f1 < 0 and tau0 < tau1; run a fixed number of iterations
    for (int i = 0; i < 30; i++) {
      if (tau1 - tau0 < 0.5e-6) break;
      float tmid = (tau0 + tau1) / 2.0;
      float fmid = root_function(osc, omega, c, pos, tmid);
      if (fmid > 0.0) tau0 = tmid; else tau1 = tmid;
    }
    return (tau0 + tau1) / 2.0;
  }

  void main()
  {
    float BETA = u_beta;
    float OMEGA = 2.0 * PI * u_freq;
    float c = abs(OMEGA) * u_vmax / BETA;

    vec2 uv = 2.0 * (gl_FragCoord.xy / u_resolution.xy) - 1.0;
    float aspect = u_resolution.x / u_resolution.y;
    uv.x *= aspect;
    uv *= u_zoom;

    float phase = OMEGA * u_time;
    vec2 osc = vec2(cos(phase), sin(phase));
    float tau = bisect_tau(osc, OMEGA, c, uv);

    vec2 osc_tau = backroll_osc(osc, OMEGA, tau);
    vec2[3] src = cubic_interp(osc_tau, OMEGA, false);

    vec2 rsrc = uv - src[0];
    float betax = src[1].x / c;
    float betay = src[1].y / c;

    float Rtau = c * tau;
    float Rmod = Rtau - betax * rsrc.x - betay * rsrc.y;

    // Encode the potential (phi, Ax, Ay) in RGB "somehow"

    float Q = 1.0 / (1.0 + Rmod);
    float Ax = betax / (1.0 + Rmod);
    float Ay = betay / (1.0 + Rmod);

    float alpha = 1.0;
    vec3 color;

    if (u_style == 0) {
      color = vec3(abs(betax), Q, abs(betay));
    } else if (u_style == 1) {
      color = vec3(0.0, Rtau / u_zoom , Q);
    } else if (u_style == 2) {
      color = vec3(Rmod / u_zoom, 0.0, 0.0);
    } else if (u_style == 3) {
      color = vec3(Q, 0.0, 0.0);
    } else {
      color = vec3(abs(Ax), 0.0, abs(Ay));
    }

    if (u_contrast != 1.0) color = pow(color, vec3(u_contrast));
    fragColor = vec4(color, alpha);
  }
`;

// Compile Shader
function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('Shader compile error:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

// Create Program
function createProgram(gl, vertexShader, fragmentShader) {
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(program));
    return null;
  }
  return program;
}

// Compile shaders and create program
const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
const program = createProgram(gl, vertexShader, fragmentShader);
if (!program) throw new Error('Program creation failed');

// Set up attributes and uniforms
const positionAttributeLocation = gl.getAttribLocation(program, 'a_position');
const resolutionUniformLocation = gl.getUniformLocation(program, 'u_resolution');
const sourceUniformLocation = gl.getUniformLocation(program, 'u_source_vertices');
const timeUniformLocation = gl.getUniformLocation(program, 'u_time');
const zoomUniformLocation = gl.getUniformLocation(program, 'u_zoom');
const contrastUniformLocation = gl.getUniformLocation(program, 'u_contrast');
const vmaxUniformLocation = gl.getUniformLocation(program, 'u_vmax');
const betaUniformLocation = gl.getUniformLocation(program, 'u_beta');
const freqUniformLocation = gl.getUniformLocation(program, 'u_freq');
const styleUniformLocation = gl.getUniformLocation(program, 'u_style');

// Create buffer for a full-screen quad
const positionBuffer = gl.createBuffer();
const positions = new Float32Array([
  -1, -1,
  1, -1,
  -1, 1,
  -1, 1,
  1, -1,
  1, 1
]);
gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

// Set up VAO
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
gl.enableVertexAttribArray(positionAttributeLocation);
gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

gl.enable(gl.BLEND);
gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

// Animation loop
const numPlotStyles = 5;
let plotStyle = 3;
const betaFPSfilter = 1.0 / 100.0;
let filteredFPS = 0.0;
let lastTime = performance.now();
let simTime = 0.0;
let zoomLevel = 8.0;
let betaLevel = 0.85;
let freqValue = 0.20;
let contrastLevel = 1.0;
let showPath = false;

function keyDownEvent(e) {
  let code = e.keyCode;
  let key = e.key;

  if (key == 'Tab') {
    plotStyle += e.shiftKey ? -1 : 1;
    if (plotStyle < 0) plotStyle = numPlotStyles - 1;
    if (plotStyle == numPlotStyles) plotStyle = 0;
    e.preventDefault();
    return;
  }

  if (key == ' ') {
    showPath = !showPath;
    e.preventDefault();
    return;
  }

  if ((key == 'b' || key == 'B')) {
    if (e.shiftKey) {
      // dial beta down
      betaLevel *= 0.95;
      if (betaLevel < 0.005) betaLevel = 0.005;
      return;
    }
    // crank beta up
    betaLevel /= 0.95;
    if (betaLevel > 0.995) betaLevel = 0.995;
    return;
  }

  if (key == 'f' || key == 'F') {
    freqValue += e.shiftKey ? -0.05 : 0.05;
    if (freqValue < 0.05) freqValue = 0.05;
    return;
  }

  if (code == 38 && e.shiftKey) { // shift + up
    zoomLevel *= 0.80;
    return;
  }

  if (code == 40 && e.shiftKey) { // shift + down
    zoomLevel *= 1.25;
    return;
  }

  if (code == 38 || code == 40) { // up/down keys cycles through the preset paths
    if (code == 38) IPATH = (IPATH + 1) % PATH_FUNCTIONS.length;
    if (code == 40) IPATH = (IPATH == 0 ? PATH_FUNCTIONS.length - 1 : IPATH - 1);
    generate_preset_path(IPATH);
    VMAX = find_maximum_velocity(1.0, sourceVertices);
    console.log([IPATH, VMAX]);
    return;
  }

  if (key == 'n' || key == 'N') {
    // Add a little bit off random noise to the current path
    for (let i = 0; i < numSourceVertices; i++) {
      sourceVertices[2 * i + 0] += (2 * Math.random() - 1) * 0.01;
      sourceVertices[2 * i + 1] += (2 * Math.random() - 1) * 0.01;
    }
    VMAX = find_maximum_velocity(1.0, sourceVertices);
    console.log(["added noise", VMAX]);
    return;
  }

  if (key == 'r' || key == 'R') {
    simTime = 0.0;
    contrastLevel = 1.0;
    return;
  }

  if (key == 'c' || key == 'C') {
    contrastLevel *= (e.shiftKey ? 1.25 : 0.80);
    console.log("contrast", contrastLevel);
    return;
  }

}

window.addEventListener('keydown', keyDownEvent);

function render() {
  const time = performance.now();
  const elapsedTimeSeconds = (time - lastTime) / 1000; // Time in seconds
  lastTime = time;
  simTime += elapsedTimeSeconds;

  if (elapsedTimeSeconds > 0.0 && elapsedTimeSeconds < 1.0)
    filteredFPS = (betaFPSfilter) * (1.0 / elapsedTimeSeconds) + (1.0 - betaFPSfilter) * filteredFPS;

  // Update uniforms
  gl.useProgram(program);
  gl.uniform2f(resolutionUniformLocation, canvasgl.width, canvasgl.height);
  gl.uniform2fv(sourceUniformLocation, sourceVertices);
  gl.uniform1f(timeUniformLocation, simTime);
  gl.uniform1f(zoomUniformLocation, zoomLevel);
  gl.uniform1f(contrastUniformLocation, contrastLevel);
  gl.uniform1f(betaUniformLocation, betaLevel);
  gl.uniform1f(vmaxUniformLocation, VMAX);
  gl.uniform1f(freqUniformLocation, freqValue);
  gl.uniform1i(styleUniformLocation, plotStyle);

  // Clear and draw
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLES, 0, 6);

  // Text overlay
  const ctx = canvas2d.getContext('2d');
  //ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas2d.width, canvas2d.height);
  ctx.globalAlpha = 1.0;
  ctx.fillStyle = 'white';
  ctx.font = '20px Arial';
  ctx.fillText('<fps> = ' + filteredFPS.toFixed(1), 20.0, canvas2d.height - 25.0);
  ctx.fillText('[b] beta = ' + betaLevel.toFixed(4) + ' [f] (anim.) freq = ' + freqValue.toFixed(4), 20.0, 25.0);

  if (showPath) {
    /*ctx.strokeStyle = 'red';
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(canvas2d.width, canvas2d.height);
    ctx.moveTo(canvas2d.width, 0);
    ctx.lineTo(0, canvas2d.height);
    ctx.stroke();*/

    ctx.fillText('Path preset #' + IPATH.toFixed(0) + ' (up/dn keys)', 20.0, 45.0);
    ctx.lineWidth = 4;
    ctx.strokeStyle = 'rgba(255,255,255,0.33)';
    ctx.beginPath();
    const wh = [canvas2d.width, canvas2d.height];
    let coords = canvas_coord([sourceVertices[0], sourceVertices[1]], wh, zoomLevel);
    ctx.moveTo(coords[0], coords[1]);
    for (let i = 1; i < numSourceVertices; i++) {
      coords = canvas_coord([sourceVertices[i * 2 + 0], sourceVertices[i * 2 + 1]], wh, zoomLevel);
      ctx.lineTo(coords[0], coords[1])
    }
    coords = canvas_coord([sourceVertices[0], sourceVertices[1]], wh, zoomLevel);
    ctx.lineTo(coords[0], coords[1]);
    ctx.stroke();
  }

  requestAnimationFrame(render);
}

// Start animation
gl.useProgram(program);
render();
