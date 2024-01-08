'use strict';

// Vertex shader
const vertexShaderSource = `
attribute vec3 vertex;
attribute vec2 texCoord;
uniform mat4 ModelViewProjectionMatrix;
uniform vec3 translateSphere;

varying vec2 texCoordV;
uniform vec3 userPoint;

uniform float scaleK;
uniform float b;

mat4 translation(vec3 t) {
    mat4 dst;

    dst[0][0] = 1.0;
    dst[0][ 1] = 0.0;
    dst[0][ 2] = 0.0;
    dst[0][ 3] = 0.0;
    dst[1][ 0] = 0.0;
    dst[1][ 1] = 1.0;
    dst[1][ 2] = 0.0;
    dst[1][ 3] = 0.0;
    dst[2][ 0] = 0.0;
    dst[2][ 1] = 0.0;
    dst[2][ 2] = 1.0;
    dst[2][ 3] = 0.0;
    dst[3][ 0] = t.x;
    dst[3][ 1] = t.y;
    dst[3][ 2] = t.z;
    dst[3][ 3] = 1.0;

    return dst;
}

mat4 scaling(float s){
    mat4 dst;

    dst[0][0] = s;
    dst[0][ 1] = 0.0;
    dst[0][ 2] = 0.0;
    dst[0][ 3] = 0.0;
    dst[1][ 0] = 0.0;
    dst[1][ 1] = s;
    dst[1][ 2] = 0.0;
    dst[1][ 3] = 0.0;
    dst[2][ 0] = 0.0;
    dst[2][ 1] = 0.0;
    dst[2][ 2] = s;
    dst[2][ 3] = 0.0;
    dst[3][ 0] = 0.0;
    dst[3][ 1] = 0.0;
    dst[3][ 2] = 0.0;
    dst[3][ 3] = 1.0;

    return dst;
}

void main() {
    vec4 tex1 = vec4(texCoord,0.,1.) * translation(userPoint);
    vec4 tex2 = tex1 * scaling(scaleK);
    vec4 tex3 = tex2 * translation(-userPoint);

    texCoordV=tex3.xy;
    vec4 vertPos4 = ModelViewProjectionMatrix * vec4(vertex, 1.0);
    vec3 vertPos = vec3(vertPos4) / vertPos4.w;

    gl_Position = ModelViewProjectionMatrix * vec4(vertex,1.0);
    if(b>0.0){
          vec4 sphere = translation(userPoint)*vec4(vertex,1.0);
          gl_Position=ModelViewProjectionMatrix*sphere;
        }
}`;


// Fragment shader
const fragmentShaderSource = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
   precision highp float;
#else
   precision mediump float;
#endif

uniform sampler2D tmu;
uniform float b;
varying vec2 texCoordV;
void main() {
    gl_FragColor = texture2D(tmu, texCoordV);
    if(b>0.){
            gl_FragColor = vec4(0.,0.,0.,1.);
        }
}`;

let gl;                         // The webgl context.
let surface;                    // A surface model
let shProgram;                  // A shader program
let spaceball;                  // A SimpleRotator object that lets the user rotate the view by mouse.
let sphere;
let userPoint = [1.0, 1.0];

document.getElementById("draw").addEventListener("click", redraw);



// Constructor
function Model(name) {
    this.name = name;
    this.iVertexBuffer = gl.createBuffer();
    this.iVertexTextureBuffer = gl.createBuffer();
    this.count = 0;

    this.BufferData = function(vertices, verticesTexture) {

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iVertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STREAM_DRAW);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iVertexTextureBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verticesTexture), gl.STREAM_DRAW);

        this.count = vertices.length/3;
    }

    this.Draw = function() {

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iVertexBuffer);
        gl.vertexAttribPointer(shProgram.iAttribVertex, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(shProgram.iAttribVertex);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iVertexTextureBuffer);
        gl.vertexAttribPointer(shProgram.iAttribVertexTexture, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(shProgram.iAttribVertexTexture);

        gl.drawArrays(gl.TRIANGLES, 0, this.count);
    }
}


// Constructor
function ShaderProgram(name, program) {

    this.name = name;
    this.prog = program;

    // Location of the attribute variable in the shader program.
    this.iAttribVertex = -1;
    // Location of the attribute variable in the shader program.
    this.iAttribNormal = -1;
    // Location of the uniform specifying a color for the primitive.
    this.iAttribVertexTexture = -1;
    this.iUserPoint = -1;
    this.iColor = -1;
    // Location of the uniform matrix representing the combined transformation.
    this.iModelViewProjectionMatrix = -1;

    this.Use = function() {
        gl.useProgram(this.prog);
    }
}


/* Draws a colored cube, along with a set of coordinate axes.
 * (Note that the use of the above drawPrimitive function is not an efficient
 * way to draw with WebGL.  Here, the geometry is so simple that it doesn't matter.)
 */
function draw() {
    gl.clearColor(0,0,0,1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    /* Set the values of the projection transformation */
    let projection = m4.perspective(Math.PI/8, 1, 8, 12);

    /* Get the view matrix from the SimpleRotator object.*/
    let modelView = spaceball.getViewMatrix();

    let rotateToPointZero = m4.axisRotation([0.707,0.707,0], 0.7);
    let translateToPointZero = m4.translation(0,0,-10);

    let matAccum0 = m4.multiply(rotateToPointZero, modelView );
    let matAccum1 = m4.multiply(translateToPointZero, matAccum0 );

    /* Multiply the projection matrix times the modelview matrix to give the
       combined transformation matrix, and send that to the shader program. */
    let modelViewProjection = m4.multiply(projection, matAccum1 );

    gl.uniformMatrix4fv(shProgram.iModelViewProjectionMatrix, false, modelViewProjection );

    /* Draw the six faces of a cube, with different colors. */
    gl.uniform3fv(shProgram.iUserPoint, [userPoint[0] / (Math.PI * 2), userPoint[1] / (Math.PI * 2), 0]);
    gl.uniform1f(shProgram.iSclAmpl, 1);

    gl.uniform3fv(shProgram.iTranslateSphere, [-0., -0., -0.])
    gl.uniform1f(shProgram.iB, -1);

    surface.Draw();
}

function CalculateVertex(v, t) {
    let a = document.getElementById("a").value;
    let b = document.getElementById("b").value;
    let c = document.getElementById("c").value;
    let d = document.getElementById("d").value;
    let m = document.getElementById("m").value;

    a = a * m, b = b * m, c = c * m, d = d * m;

    let f = a * b / Math.sqrt((a * a * Math.sin(v) * Math.sin(v) + b * b * Math.cos(v) * Math.cos(v)));
    let x = 0.5 * (f * (1 + Math.cos(t)) + (d * d - c * c) * ((1 - Math.cos(t)) / f)) * Math.cos(v);
    let y = 0.5 * (f * (1 + Math.cos(t)) + (d * d - c * c) * ((1 - Math.cos(t)) / f)) * Math.sin(v);
    let z = 0.5 * (f - ((d * d - c * c) / f)) * Math.sin(t);

    return([x,y,z]);
}

function map(val, f1, t1, f2, t2) {
    let m;
    m = (val - f1) * (t2 - f2) / (t1 - f1) + f2
    return Math.min(Math.max(m, f2), t2);
}

function CreateSurfaceData() {
    let v_end_pi = document.getElementById("v_end_pi").value;
    let t_end_pi = document.getElementById("t_end_pi").value;
    let INC = 0.1;

    let vertexList = [];
    let normalList = [];
    let vertexTextureList = [];

    for (let v = 0; v <= v_end_pi * Math.PI; v += 0.1) {
        for (let t = 0; t <= t_end_pi * Math.PI; t += 0.1) {

            let vertex1 = CalculateVertex(v, t);
            let vertex2 = CalculateVertex(v, t + 0.1);
            let vertex3 = CalculateVertex(v + 0.1, t);
            let vertex4 = CalculateVertex(v + 0.1, t + 0.1);

            let u1 = map(v, 0, v_end_pi * Math.PI, 0, 1)
            let v1 = map(t, 0, t_end_pi * Math.PI, 0, 1)
            vertexTextureList.push(u1, v1)

            u1 = map(t + INC, 0, v_end_pi * Math.PI, 0, 1)
            vertexTextureList.push(u1, v1)

            u1 = map(t, 0, v_end_pi * Math.PI, 0, 1)
            v1 = map(v + INC, 0, t_end_pi * Math.PI, 0, 1)
            vertexTextureList.push(u1, v1)

            u1 = map(t + INC, 0, v_end_pi * Math.PI, 0, 1)
            v1 = map(v, 0, t_end_pi * Math.PI, 0, 1)
            vertexTextureList.push(u1, v1)

            v1 = map(v + INC, 0, t_end_pi * Math.PI, 0, 1)
            vertexTextureList.push(u1, v1)

            u1 = map(t, 0, v_end_pi * Math.PI, 0, 1)
            v1 = map(v + INC, 0, t_end_pi * Math.PI, 0, 1)
            vertexTextureList.push(u1, v1)

            vertexList.push(...vertex1, ...vertex2, ...vertex3, ...vertex3, ...vertex2, ...vertex4);
        }
    }

    for  (let t = 0; t <= t_end_pi * Math.PI; t += 0.1){
        for (let v = 0; v <= v_end_pi * Math.PI; v += 0.1) {
            let vertex1 = CalculateVertex(v, t);
            let vertex2 = CalculateVertex(v, t + 0.1);
            let vertex3 = CalculateVertex(v + 0.1, t);
            let vertex4 = CalculateVertex(v + 0.1, t + 0.1);

            let u1 = map(t, 0, t_end_pi * Math.PI, 0, 1)
            let v1 = map(v, 0, v_end_pi * Math.PI, 0, 1)
            vertexTextureList.push(u1, v1)

            u1 = map(t + INC, 0, t_end_pi * Math.PI, 0, 1)
            vertexTextureList.push(u1, v1)

            u1 = map(t, 0, t_end_pi * Math.PI, 0, 1)
            v1 = map(v + INC, 0, v_end_pi * Math.PI, 0, 1)
            vertexTextureList.push(u1, v1)

            u1 = map(t + INC, 0, t_end_pi * Math.PI, 0, 1)
            v1 = map(v, 0, v_end_pi * Math.PI, 0, 1)
            vertexTextureList.push(u1, v1)

            v1 = map(v + INC, 0, 150, 0, 1)
            vertexTextureList.push(u1, v1)

            u1 = map(t, 0, t_end_pi * Math.PI, 0, 1)
            v1 = map(v + INC, 0, v_end_pi * Math.PI, 0, 1)
            vertexTextureList.push(u1, v1)

            vertexList.push(...vertex1, ...vertex2, ...vertex3, ...vertex3, ...vertex2, ...vertex4);
        }
    }

    return [vertexList, vertexTextureList];
}

function CalculateVertexSphere(theta, phi, radius) {
    let x = radius * Math.sin(phi) * Math.cos(theta);
    let y = radius * Math.sin(phi) * Math.sin(theta);
    let z = radius * Math.cos(phi);

    return [x, y, z];
}
function CreateSphereSurface() {
    let radius = 0.05; // Радіус сфери

    let vertexList = [];

    for (let phi = 0; phi <= Math.PI; phi += 0.1) {
        for (let theta = 0; theta <= 2 * Math.PI; theta += 0.1) {
            let vertex1 = CalculateVertexSphere(theta, phi, radius);
            let vertex2 = CalculateVertexSphere(theta, phi + 0.1, radius);
            let vertex3 = CalculateVertexSphere(theta + 0.1, phi, radius);
            let vertex4 = CalculateVertexSphere(theta + 0.1, phi + 0.1, radius);

            vertexList.push(...vertex1, ...vertex2, ...vertex3, ...vertex3, ...vertex2, ...vertex4);
        }
    }

    return [vertexList, vertexList];
}


/* Initialize the WebGL context. Called from init() */
function initGL() {
    let prog = createProgram( gl, vertexShaderSource, fragmentShaderSource );

    shProgram = new ShaderProgram('Basic', prog);
    shProgram.Use();

    shProgram.iAttribVertex              = gl.getAttribLocation(prog, 'vertex');
    shProgram.iAttribVertexTexture       = gl.getAttribLocation(prog, 'texCoord');
    shProgram.iModelViewProjectionMatrix = gl.getUniformLocation(prog,'ModelViewProjectionMatrix');
    shProgram.iUserPoint              = gl.getUniformLocation(prog, 'userPoint');
    shProgram.iTranslateSphere           = gl.getUniformLocation(prog, 'translateSphere');
    shProgram.iB                         = gl.getUniformLocation(prog, 'b');
    shProgram.iSclAmpl                   = gl.getUniformLocation(prog, 'scaleK');

    LoadTexture()
    surface = new Model('Surface');
    surface.BufferData(...CreateSurfaceData());

    //sphere = new Model('Sphere');
    //sphere.BufferData(CreateSphereSurface())

    gl.enable(gl.DEPTH_TEST);
}


/* Creates a program for use in the WebGL context gl, and returns the
 * identifier for that program.  If an error occurs while compiling or
 * linking the program, an exception of type Error is thrown.  The error
 * string contains the compilation or linking error.  If no error occurs,
 * the program identifier is the return value of the function.
 * The second and third parameters are strings that contain the
 * source code for the vertex shader and for the fragment shader.
 */
function createProgram(gl, vShader, fShader) {
    let vsh = gl.createShader( gl.VERTEX_SHADER );
    gl.shaderSource(vsh,vShader);
    gl.compileShader(vsh);
    if ( ! gl.getShaderParameter(vsh, gl.COMPILE_STATUS) ) {
        throw new Error("Error in vertex shader:  " + gl.getShaderInfoLog(vsh));
    }
    let fsh = gl.createShader( gl.FRAGMENT_SHADER );
    gl.shaderSource(fsh, fShader);
    gl.compileShader(fsh);
    if ( ! gl.getShaderParameter(fsh, gl.COMPILE_STATUS) ) {
        throw new Error("Error in fragment shader:  " + gl.getShaderInfoLog(fsh));
    }
    let prog = gl.createProgram();
    gl.attachShader(prog,vsh);
    gl.attachShader(prog, fsh);
    gl.linkProgram(prog);
    if ( ! gl.getProgramParameter( prog, gl.LINK_STATUS) ) {
        throw new Error("Link error in program:  " + gl.getProgramInfoLog(prog));
    }
    return prog;
}


/**
 * initialization function that will be called when the page has loaded
 */
function init() {
    let canvas;
    try {
        canvas = document.getElementById("webglcanvas");
        gl = canvas.getContext("webgl");
        if (!gl) {
            throw "Browser does not support WebGL";
        }
    } catch (e) {
        document.getElementById("canvas-holder").innerHTML =
            "<p>Sorry, could not get a WebGL graphics context.</p>";
        return;
    }
    try {
        initGL(true);  // initialize the WebGL graphics context
        spaceball = new TrackballRotator(canvas, draw, 0);
    } catch (e) {
        document.getElementById("canvas-holder").innerHTML =
            "<p>Sorry, could not initialize the WebGL graphics context: " + e + "</p>";
        return;
    }

    draw();
}


function redraw() {
    CreateSurfaceData()
    init()
}

window.onkeydown = (e) => {
    if (e.keyCode == 87) {
        userPoint[0] = Math.min(userPoint[0] + 0.1, Math.PI * 2);
    }
    else if (e.keyCode == 83) {
        userPoint[0] = Math.max(userPoint[0] - 0.1, 0);
    }
    else if (e.keyCode == 68) {
        userPoint[1] = Math.min(userPoint[1] + 0.1, 2 * Math.PI);
    }
    else if (e.keyCode == 65) {
        userPoint[1] = Math.max(userPoint[1] - 0.1, 0);
    }
}

function LoadTexture() {
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    const image = new Image();
    image.crossOrigin = 'anonymus';

    image.src = "texture.jpg";
    image.onload = () => {
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            image
        );
        draw()
    }
}