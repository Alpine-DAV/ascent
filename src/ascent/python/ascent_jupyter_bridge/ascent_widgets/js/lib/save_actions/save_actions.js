define(['@jupyter-widgets/base', 'text!./saveactions1.html', 'text!./saveactions1.css'],  function(widgets, template, styles) {

    var SaveActionsView = widgets.DOMWidgetView.extend({

        // Render the view.
        render: function() {
            SaveActionsView.__super__.render.apply(this, arguments);

            var styleSheet = document.createElement("style");
            styleSheet.type = "text/css";
            styleSheet.innerText = styles;
            //TODO encapsulate this
            document.head.appendChild(styleSheet);

            this.main_container = document.createElement("div");
            this.main_container.innerHTML = template;

            var that = this;
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

            // JavaScript -> Python update
            this.filename_input = this.main_container.querySelector("#filenameInput");
            this.filename_input.onchange = this.filename_changed.bind(this);

            // Python -> JavaScript update
            this.model.on('change:status', this.status_changed, this);
        },

        filename_changed: function() {
            /*
            console.log("Filename: " + this.filename_input.value);
            this.model.set('filename', this.filename_input.value);
            this.model.save_changes();
            this.touch();
            */
            this.send({event: 'filename_changed', filename: this.filename_input.value});
        },

        status_changed: function() {
            this.main_container.querySelector("#saveStatus").innerHTML = this.model.get('status');
        },

        disabled_changed: function() {
            this.email_input.disabled = this.model.get('disabled');
        },

    });

    return {SaveActionsView: SaveActionsView};
});
