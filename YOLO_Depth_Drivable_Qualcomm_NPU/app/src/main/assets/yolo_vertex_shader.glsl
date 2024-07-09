#version 320 es

precision mediump float;

in vec4 box_position;

void main() {

    gl_Position = vec4(box_position);

}
