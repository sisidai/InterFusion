<!-- ## This document summarizes potential issues and solutions that may arise during environment installation. If you encounter any issues not mentioned in this document, please submit an issue. -->


### ImportError: cannot import name 'packaging' from 'pkg_resources'
```
pip install setuptools==69.5.1
```

### ERROR: No matching distribution found for simple-romp==1.0.4
```
pip install pip==24.0
```

### ImportError: cannot import name 'OSMesaCreateContextAttribs' from 'OpenGL.osmesa'
```
pip install --upgrade pyopengl==3.1.4
```

### ValueError: Invalid device ID (0)
Insert `os.environ['PYOPENGL_PLATFORM'] = 'osmesa'` after line129 `def _create(self):` in the file `anaconda3/envs/interfusion_s1/lib/python3.8/site-packages/pyrender/offscreen.py` to enforce the use of osmesa. Then it should look like:
```
    def _create(self):                                                             
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'                                  
        if 'PYOPENGL_PLATFORM' not in os.environ:                                   
            from pyrender.platforms.pyglet_platform import PygletPlatform           
            self._platform = PygletPlatform(self.viewport_width,                    
                                            self.viewport_height)                   
```