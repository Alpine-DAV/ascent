//TODO figure out how to compile requirejs
//TODO use shadow dom or iframes for encapsulation
//TODO use display(HTML(filename="trackball.css.html")) for separate HTML/CSS?

//require.undef('trackball')

require.config({
  //Define 3rd party plugins dependencies
  paths: {
    three: "https://threejs.org/build/three.min",
    TrackballControls: "https://rawgit.com/mrdoob/three.js/master/examples/js/controls/TrackballControls"
  },
  shim: {
    'TrackballControls': {
      deps: ['three'],
      exports: 'THREE.TrackballControls'
    }
  },
  map: {
        '*': {
            three: 'three-glue'
        },
        'three-glue': {
            three: 'three'
        }
    }
});


define('three-glue', ['three'], function (three) {
    window.THREE = three;
    return three;
});

define('trackball', ['@jupyter-widgets/base', 'three', 'TrackballControls'], function(widgets, THREE) {       
    var trackball_view = widgets.DOMWidgetView.extend({
        render: function() {
            var that = this;
            
            this.move_trackball = false;
            
            this.main_container = document.createElement("div");
            this.main_container.tabIndex = 1;
            this.main_container.style.background = "white";
            this.main_container.style.overflow = "hidden";
            
            var styles = `
                .col {
                    display: inline-block;
                }
                #controlsDiv {
                    display: flex;
                    align-items: stretch;
                    text-align: center;
                }
                #trackballDiv {
                    flex-grow: 1;
                }
                #buttonsDiv {
                    flex-grow: 1;
                }
                #canvasDiv {
                    text-align: center;
                    margin-top: 2rem;
                }
                .row {
                    margin-bottom: 2rem;
                }
                #simControl {
                    text-align: right;
                    margin-top: 5rem;
                }
            `;
            var styleSheet = document.createElement("style");
            styleSheet.type = "text/css";
            styleSheet.innerText = styles;
            this.el.appendChild(styleSheet);
            
            this.main_container.innerHTML = `
                <div id="canvasDiv" class="row"></div>
                <div id="controlsDiv" class="row">
                    <div id="trackballDiv" class="col"></div>
                    <div id="buttonsDiv" class="col">
                        <div id="panButtons" class="row">
                            <button id="move_up" class="control-button">Move up (W)</button>
                            <div id="lowerButtons">
                                <button id="move_left" class="control-button">Move left (A)</button>
                                <button id="move_down" class="control-button">Move down (S)</button>
                                <button id="move_right" class="control-button">Move right (D)</button>                            
                            </div>
                        </div>
                        <div id="fbButtons" class="row">
                            <button id="move_forward" class="control-button">Move forward</button>
                            <button id="move_back" class="control-button">Move back</button>
                        </div>
                        <div id="rollButtons" class="row">
                            <button id="roll_c" class="control-button">Roll clockwise</button>
                            <button id="roll_cc" class="control-button">Roll counterclockwise</button>
                        </div>
                        <div id="pitchButtons" class="row">
                            <button id="pitch_up" class="control-button">Pitch up</button>
                            <button id="pitch_down" class="control-button">Pitch down</button>
                        </div>
                        <div id="yawButtons" class="row">
                            <button id="yaw_right" class="control-button">Yaw right</button>
                            <button id="yaw_left" class="control-button">Yaw left</button>
                        </div>
                        <div id="simControl" class="row">
                            <button id="next" class="control-button">Advance Simulation</button>
                        </div>
                    </div>
                </div>
            `;
            this.main_container.querySelectorAll('.control-button').forEach((button) => {
                // add css classes to make buttons look like jupyter widget buttons
                const cls = ["p-Widget", "jupyter-widgets", "jupyter-button", "widget-button"];
                button.classList.add(...cls);
                //send button click to kernel
                button.addEventListener("click", () => {
                    that.send({event: 'button', code: button.id});
                });
            });
            this.el.appendChild(this.main_container);


            this.canvas = document.createElement('canvas');
            this.context = this.canvas.getContext('2d');
            this.canvas.width = this.model.get('width');
            this.canvas.height = this.model.get('height');
                  
            this.main_container.querySelector("#canvasDiv").appendChild(this.canvas);
            
            this.model.on('change:image', this.update_image, this);
            this.update_image();
            
            this.model.on('change:scene_bounds', this.update_scene_bounds, this);
            
            this.model.on('change:camera_info', this.update_camera, this);
            
            this.init_trackball();
            this.animate_trackball();
        },
        events: {
            'keydown': 'keydown',
        },
        keydown: function(e) {
            var code = e.keyCode || e.which;
            this.send({event: 'keydown', code: code});
        },
        update_image: function() {
            let buffer = this.model.get('image');
            var img = new Image;
            img.src = "data:image/png;base64," + buffer;
            var that = this;
            img.onload = function() {
                that.context.clearRect(0, 0, that.canvas.width, that.canvas.height);
                that.context.drawImage(img, 0, 0, that.canvas.width, that.canvas.height);
            };
        },
        init_trackball: function () {
            this.camera = new THREE.PerspectiveCamera(this.model.get('camera_info')['fov'],
                                                      this.model.get('width') / this.model.get('height'),
                                                      1,
                                                      1000);
            this.update_camera();
                              
            this.renderer = new THREE.WebGLRenderer();
            this.renderer.setSize(this.model.get('width') / 2, this.model.get('height') / 2);
            this.renderer.domElement.tabIndex = 1;
            
            var that = this;
            this.renderer.domElement.addEventListener('mousedown', function(e) {
                that.move_trackball = true;
            });

            document.addEventListener('mouseup', function(e) {
                if(that.move_trackball) {
                    var camera_info = {}
                    camera_info['position'] = [that.camera.position.x,
                                               that.camera.position.y,
                                               that.camera.position.z];

                    var look_at = that.controls.target; //TODO do we need get_look_at?
                    camera_info['look_at'] = [look_at.x, look_at.y, look_at.z];

                    camera_info['up'] = [that.camera.up.x,
                                         that.camera.up.y,
                                         that.camera.up.z];

                    camera_info['fov'] = that.camera.fov;

                    that.send({event: 'mouseup', camera_info: camera_info});
                    
                    
                    that.move_trackball = false;
                }
            });
            this.main_container.querySelector("#trackballDiv").appendChild(this.renderer.domElement);
            
            this.controls = new THREE.TrackballControls(this.camera, this.renderer.domElement);
            this.controls.rotateSpeed = 0.5;
                
            this.scene = new THREE.Scene();
            var geometry = this.generate_cube_geometry();
            
            var material = new THREE.MeshNormalMaterial();
            var mesh = new THREE.Mesh(geometry, material);
            this.scene.add(mesh);
        },     
        animate_trackball: function() {
            requestAnimationFrame( this.animate_trackball.bind(this) );
            this.controls.handleResize()
            this.controls.update();
            this.render_trackball();
        },
        render_trackball: function() {
            this.renderer.render( this.scene, this.camera );
        },
        //get look_at point
        get_look_at: function() {
            //model's (not updated) current camera info
            var camera_info = this.model.get('camera_info');
            var position = camera_info['position'];
            var look_at = camera_info['look_at'];
            
            //construct threejs versions
            var position_vector = new THREE.Vector3(position[0], position[1], position[2]);
            var look_at_vector = new THREE.Vector3(look_at[0], look_at[1], look_at[2]);
            
            //try to keep the distance from the look_at point consistent
            var magnitude = position_vector.distanceTo(look_at_vector);
            
            //use the camera's rotation matrix to find new look_at direction
            var vector = new THREE.Vector3(0, 0, -magnitude);
            vector.applyEuler(this.camera.rotation, this.camera.rotation.order);
            
            //translate to the camera position
            vector.add(position_vector);
            
            return vector;
        },
        //set camera's position, look_at, up
        update_camera: function() {
            var camera_info = this.model.get('camera_info');
            var position = camera_info['position'];
            var look_at = camera_info['look_at'];
            var up = camera_info['up'];
            var fov = camera_info['fov'];

            this.camera.position.set(position[0], position[1], position[2]);
            this.camera.up.set(up[0], up[1], up[2]);

            if(this.controls != undefined){
                this.controls.target.set(look_at[0], look_at[1], look_at[2]);
            }
            
            //only update projection matrix when nessecary in case it's expensive
            if(this.camera.fov != fov) {
                this.camera.fov = fov;
                this.camera.updateProjectionMatrix();
            }
            
        },
        //update that threejs cube's position and size to match ascent's bounding box
        update_scene_bounds: function() {
            console.log("TODO: Change Scene Bounds")
        },
        //generate cube geometry based on ascent's bounding box
        generate_cube_geometry: function() {
            var scene_bounds = this.model.get('scene_bounds')
            
            var x_length = scene_bounds[3] - scene_bounds[0]
            var y_length = scene_bounds[4] - scene_bounds[1]
            var z_length = scene_bounds[5] - scene_bounds[2]
            
            var geometry = new THREE.CubeGeometry(x_length, y_length, z_length);
            
            //translate to the right position by aligining "bottom left" corners
            geometry.computeBoundingBox()
            geometry.translate(scene_bounds[0] - geometry.boundingBox.min.x,
                               scene_bounds[1] - geometry.boundingBox.min.y,
                               scene_bounds[2] - geometry.boundingBox.min.z)
            
            //set rotation point for mouse controls
            //TODO use look_at instead of center
            geometry.computeBoundingBox()
            var center = geometry.boundingBox.getCenter()
            this.controls.target.set(center.x, center.y, center.z);

            return geometry
        }
    });
        
    return {TrackballView: trackball_view};
});
