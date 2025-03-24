#version 320 es
precision mediump float;
out vec4 fragColor;
uniform vec4 box_color;

void main() {
    fragColor = vec4(box_color); 
}
