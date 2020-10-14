/*
 * Frame Retention for Video Recognition
 *
 * Inspired by:
 *
 * Keras-Android-XOR
 * https://github.com/OmarAflak/Keras-Android-XOR
 *
 * CNN-models
 * https://github.com/km1414/CNN-models
 */

package aflak.me.tensorflowlitexor;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.ImageReader;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.support.annotation.RequiresApi;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.util.SparseIntArray;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Arrays;

public class CoInferenceActivity extends AppCompatActivity {
    private TensorFlowInferenceInterface inferenceInterface;

    private ServerSocket serverSocket;
    Handler UIHandler;
    Thread Thread1 = null;
    /** 设置端口以接收TCP消息. */
    public static final int SERVERPORT = 8000;

    /** UI控件 */
    private TextView tvPicture;
    private TextView tvInternet;
    private TextView tvInference;
    private TextView tvSsid;
    private TextView tvOffInference;
    private TextView tvOnline;
    private TextView tvOffline;
    private TextView btnInference;

    private Bitmap selectedImage;

    /**
     * Camera 变量
     */
    private static final String TAG = "AndroidCamera";
    private TextureView textureView;
    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();
    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 90);
        ORIENTATIONS.append(Surface.ROTATION_90, 0);
        ORIENTATIONS.append(Surface.ROTATION_180, 270);
        ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }
    private String cameraId;
    protected CameraDevice cameraDevice;
    protected CameraCaptureSession cameraCaptureSessions;
    protected CaptureRequest.Builder captureRequestBuilder;
    private android.util.Size imageDimension;
    private ImageReader imageReader;
    private static final int REQUEST_CAMERA_PERMISSION = 200;
    /** onResume 和onPause 控制的线程和Handler */
    private Handler mBackgroundHandler;
    private HandlerThread mBackgroundThread;

    /** 运行状态控制 */
    private final Object lock = new Object(); // 新建一个对象作为锁
    private boolean runClassifier = false;    // 推断执行状态变量
    private HandlerThread backgroundThread;
    private Handler backgroundHandler;

    /** 时间统计 */
    private long lastSend = 0; // 发送中间结果的时间
    private long lastRecv = 0; // 接收最后结果的时间

    private String ssid = null;


    static {
        OpenCVLoader.initDebug();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // UI布局
        setContentView(R.layout.activity_co_inference);

        /**
         * 找到views并启动Listener
         */
        tvPicture = findViewById(R.id.samp); // Sampling and pre-pfocessing 的值
        tvInference = findViewById(R.id.oninf); // Local time 的值
        tvInternet = findViewById(R.id.server); // Server time 的值
        tvSsid = findViewById(R.id.apid); // APID: UniTest_的值  => ssid 初始化为B
        tvOffInference = findViewById(R.id.offinf); // Time 的值
        btnInference = findViewById(R.id.button); // 视频中的按钮
        tvOnline = findViewById(R.id.online);   // Online Mode
        tvOffline = findViewById(R.id.offline); // Offline Mode
        textureView = findViewById(R.id.texture); // 视频画面  => TextureView 是一个由于显示数据流的UI控件
        assert textureView != null;
        textureView.setSurfaceTextureListener(textureListener);
        initListener();// 启动视频中按钮的监听器

        /**
         * 加载冻结的 tensorflow 模型
         */
        Log.d(getClass().getSimpleName(), "Loading Model");
//        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), "frozen_googlenet.pb");
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), "frozen_VGG.pb");
        Log.d(getClass().getSimpleName(), "Loaded Model");

        /**
         * 启动 TCP Receiver
         */
        UIHandler = new Handler(); // Handler可以当成子线程与主线程的消息传送的纽带,子线程可以通过Handler来将UI更新操作切换到主线程中执行
        /** 启动监听远程连接的线程，新来的连接新开一个线程Thread2去处理，然后退出.(在Thread2中处理了新来的数据后，又会新建一个Thread1继续监听) */
        this.Thread1 = new Thread(new Thread1());
        this.Thread1.start();

    }

    private void initListener() {
        // 视频中按钮的点击事件监听
        btnInference.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(runClassifier){
                    // 点击开始按钮
                    btnInference.setText("\uf144");
                    stopGeneralBackgroundThread(); // 停止主线程以进行周期性推断
                }else {
                    // 点击暂停按钮
                    btnInference.setText("\uf28d");
                    startGeneralBackgroundThread(); // 启动主线程以进行周期性推断
                }
            }
        });
    }

    /** 本地推理整个模型. */
    private float[] predict(float[] input){
        float output[] = new float[10];
        // step1: feed 数据
        inferenceInterface.feed("input_1", input, 1, 32,32,3);
        // step2: 执行推理
        inferenceInterface.run(new String[]{"predictions/Softmax"});
        // step3: 获取推理结果
        inferenceInterface.fetch("predictions/Softmax", output);

        return output;  // return prediction
    }

    /** 返回部分推理结果. */
    private float[] predictCut(float[] input){
        float output[] = new float[65536];

        // step1: feed 数据
        inferenceInterface.feed("input_1", input, 1, 32,32,3);
        // step2: 执行部分推理
        inferenceInterface.run(new String[]{"concatenate_1/concat"});
        // step3: 获取推理结果
        inferenceInterface.fetch("concatenate_1/concat", output);

        return output;  // return intermediate result
    }

    /**
     * 启动主线程以进行周期性推断
     */
    private void



    startGeneralBackgroundThread() {
        // 步骤1：创建HandlerThread实例对象 => 新的工作线程: 继承Thread类 & 封装Handler类
        backgroundThread = new HandlerThread("ClassifierBG");
        // 步骤2：启动线程
        backgroundThread.start();
        // 步骤3：创建工作线程Handler => 关联HandlerThread的Looper对象、实现消息处理操作 & 与其他线程进行通信
        backgroundHandler = new Handler(backgroundThread.getLooper());
        // 步骤4：修改推断执行状态变量
        synchronized (lock) {
            runClassifier = true;
        }
        // 步骤5：使用工作线程Handler向工作线程的消息队列发送消息 => 在工作线程中，当消息循环时取出对应消息 & 在工作线程执行相关操作
        backgroundHandler.post(periodicClassify);
        /**
         * 首先，post和postDelay都是Handler的方法，用以在子线程中发送Runnable对象的方法；
         *
         * 其次，Android中post()方法可以直接在非UI线程中更新UI，不同与Handelr的Send类方法，需要进行切换；
         *
         * 最后，两个方法在实现UI线程事件的时间上有所区别，postDelayed()方法用以延期执行，post则是立即执行；
         * */
    }

    /**
     * 停止周期性推断的主线程
     */
    @RequiresApi(api = Build.VERSION_CODES.JELLY_BEAN_MR2)
    private void stopGeneralBackgroundThread() {
        backgroundThread.quitSafely();
        try {
            backgroundThread.join();
            backgroundThread = null;
            backgroundHandler = null;
            synchronized (lock) {
                runClassifier = false;
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * 主线程 runnable
     * 推理 15帧/秒
     */
    private Runnable periodicClassify =
            new Runnable() {
                @Override
                public void run() {
                    int timecost = 0;
                    // 同步锁
                    synchronized (lock) {
                        // 推理状态为运行
                        if (runClassifier) {
                            timecost = takeBitmap();
                        }
                    }
                    int delay = 67 - (timecost % 67);
                    try{
                        int j = (int)(timecost / 67);
                        for(int i = 0; i < j; i++){
                            /**
                             * "INFO0444" 表示错过了一帧.
                             */
                            Log.i("INFO0444", Integer.toString(j));
                        }

                    }catch(Exception e){

                    }
                    backgroundHandler.postDelayed(periodicClassify, delay);
                }
            };

    /**
     * 抓取一张照片，检查互联网连接并执行推理.
     * 返回完成所有这些操作所需的时间.
     */
    public int takeBitmap(){

        // t0 表示整体开始的时间
        long t0 = System.currentTimeMillis();
        long t1 = 0;

        // step1: 选中一张照片
        selectedImage = textureView.getBitmap();

        try {
            // step2: 将照片转换成float数组
            float[] inputData = getInputData(selectedImage);

            t1 = System.currentTimeMillis();
            // 设置 Sampling and pre-pfocessing的值  => 数据获取时间
//            tvPicture.setText(String.valueOf(t1 - t0));
            final long data_fetch_time = t1 - t0;
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    tvPicture.setText(String.valueOf(data_fetch_time));
                }
            });
            long t2 = t1 - t0;

            float[] output;

            // step3: 检查网络是否连接
            boolean net = isConnected();
            // 网络连接可用
            if(net){
                if(ssid == null){ // 初始ssid为NULL
//                    tvSsid.setText("NULL");
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            tvSsid.setText("NULL");
                        }
                    });
                }else{
//                    tvSsid.setText(getCurrentBssid());
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            tvSsid.setText(getCurrentBssid());
                        }
                    });
                }

                if(ssid != null && getCurrentBssid().equals(ssid)){
                    long currentT = System.currentTimeMillis();
                    // 距离上次接收超过150 => 设置网络连接为False
                    if(currentT - lastRecv > 150){
                        net = false;
                    }
                    // 距离上次接收超过1500
                    if(currentT - lastRecv > 1500){
                        disconnect(); // 断开与当前wifi的连接，然后让系统重新连接.
                        lastRecv = currentT + 10000;
                    }
                }else{
                    // 设置ssid为当前连接的ssid
                    ssid = getCurrentBssid();
                    lastRecv = System.currentTimeMillis();
                }
            }

            // step4: 根据网络连接开始在线 or 离线推理
            if(net){ // 网络连接可用
                // 步骤1：将在线模式设为红色、离线模式设为灰色
//                tvOnline.setTextColor(Color.RED);
//                tvOffline.setTextColor(Color.GRAY);
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        tvOnline.setTextColor(Color.RED);
                        tvOffline.setTextColor(Color.GRAY);
                    }
                });
                // 步骤2：更新t0 => 本地部分推理开始时间
                t0 = System.currentTimeMillis();

                // 步骤3：本地推理部分模型
                output = predictCut(inputData);

                // 步骤4：更新t1 => 本地部分推理结束时间
                t1 = System.currentTimeMillis();

                // 步骤5：更新界面中Local time 的值
//                tvInference.setText(String.valueOf(t1 - t0));
                final long localPartInferTime = t1 - t0;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        tvInference.setText(String.valueOf(localPartInferTime));
                    }
                });
                /**
                 * "INFO0111" 表示推理时间
                 * 当 "ONLINE" 时，它将在设备上执行部分推理，并将其余部分发送到边缘进行推理
                 * 当 "OFFLINE" 时，它将在设备上执行整个推理
                 */
                Log.i("INFO0111 ONLINE " , String.valueOf(t1 - t0 + t2));
                // 步骤6：更新 lastSend 的值
                lastSend = System.currentTimeMillis();
                // 步骤7：通过TCP发送中间结果
                message(output);

            }else{ // 网络连接不可用
                // 步骤1：将在线模式设为灰色、离线模式设为红色
//                tvOnline.setTextColor(Color.GRAY);
//                tvOffline.setTextColor(Color.RED);
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        tvOnline.setTextColor(Color.GRAY);
                        tvOffline.setTextColor(Color.RED);
                    }
                });
                // 步骤2：更新t0 => 本地推理开始时间
                t0 = System.currentTimeMillis();

                // 步骤3：本地推理整个模型
                output = predict(inputData);

                // 步骤4：更新t1 => 本地推理结束时间
                t1 = System.currentTimeMillis();
                // 步骤5：更新界面中Time的值
//                tvOffInference.setText(String.valueOf(t1 - t0));
                final long localInferTime = t1 - t0;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        tvOffInference.setText(String.valueOf(localInferTime));
                    }
                });
                // 日志：INFO0111 OFFLINE = 本地推理时间 + 数据获取时间
                Log.i("INFO0111 OFFLINE " , String.valueOf(t1 - t0 + t2));
            }

        } catch (Exception e) {
            e.printStackTrace();
            Log.d(getClass().getSimpleName(),"Failure: " + e);
        }

        // 完成的时间
        long te = System.currentTimeMillis();
        return (int)(te - t0);
    }

    /**
     * Camera API functions
     */
    TextureView.SurfaceTextureListener textureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            openCamera();
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            return false;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        }
    };

    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice camera) {
            //当相机打开成功之后会回调此方法
            //一般在此进行获取一个全局的CameraDevice实例，开启相机预览等操作
            Log.e(TAG, "onOpened");
            cameraDevice = camera;//获取CameraDevice实例
            createCameraPreview();//创建相机预览会话
        }

        @Override
        public void onDisconnected(CameraDevice camera) {
            //相机设备失去连接(不能继续使用)时回调此方法，同时当打开相机失败时也会调用此方法而不会调用onOpened()
            //可在此关闭相机，清除CameraDevice引用
            cameraDevice.close();
            cameraDevice = null;
        }

        @Override
        public void onError(CameraDevice camera, int error) {

            //相机发生错误时调用此方法
            cameraDevice.close();
            cameraDevice = null;
        }
    };

    /**
     * onResume中调用，开启一个工作线程mBackgroundThread
     */
    protected void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("Camera Background");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    /**
     * onPause中调用，停止一个工作线程mBackgroundThread
     */
    protected void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * 创建相机预览会话
     */
    protected void createCameraPreview() {
        try {
            //获取SurfaceTexture
            SurfaceTexture texture = textureView.getSurfaceTexture();
            assert texture != null;
            //设置SurfaceTexture大小
            texture.setDefaultBufferSize(imageDimension.getWidth(), imageDimension.getHeight());
            //获取预览的surface
            Surface surface = new Surface(texture);
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(surface);
            cameraDevice.createCaptureSession(Arrays.asList(surface), new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                    //相机已经关闭
                    if (null == cameraDevice) {
                        return;
                    }
                    // session准备就绪后，我们开始显示预览.
                    cameraCaptureSessions = cameraCaptureSession;
                    updatePreview();
                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
                    Toast.makeText(CoInferenceActivity.this, "Configuration change", Toast.LENGTH_SHORT).show();
                }
            }, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    /** 打开相机 */
    private void openCamera() {
        // 1. 相机管理器
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        Log.e(TAG, "is camera open");
        try {
            // 2. 拿到相机id
            cameraId = manager.getCameraIdList()[0];
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            // 3. 相机配置
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            assert map != null;
            imageDimension = map.getOutputSizes(SurfaceTexture.class)[0];
            // 4. 为相机添加权限，并让用户授予权限
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(CoInferenceActivity.this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
                return;
            }
            // 5. 打开相机
            manager.openCamera(cameraId, stateCallback, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
        Log.e(TAG, "openCamera X");
    }

    protected void updatePreview() {
        if (null == cameraDevice) {
            Log.e(TAG, "updatePreview error, return");
        }
        captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
        try {
            cameraCaptureSessions.setRepeatingRequest(captureRequestBuilder.build(), null, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    /**
     * 关闭相机
     */
    private void closeCamera() {
        if (null != cameraDevice) {
            cameraDevice.close();
            cameraDevice = null;
        }
        if (null != imageReader) {
            imageReader.close();
            imageReader = null;
        }
    }

    /** 如果权限被拒绝. */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                // close the app
                Toast.makeText(CoInferenceActivity.this, "Sorry!!!, you can't use this app without granting permission", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    @Override
    protected void onResume() {
        /**
         * 通常是当前的acitivty被暂停了，比如被另一个透明或者Dialog样式的Activity覆盖了），之后dialog取消，
         * activity回到可交互状态，调用onResume()
         */
        super.onResume();
        Log.e(TAG, "onResume");

        if (textureView.isAvailable()) {
            openCamera();
        } else {
            textureView.setSurfaceTextureListener(textureListener);
        }
        startBackgroundThread();
    }

    @Override
    protected void onPause() {
        /**
         * Paused状态：当Activity被另一个透明或者Dialog样式的Activity覆盖时的状态。
         * 此时它依然与窗口管理器保持连接，系统继续维护其内部状态，它仍然可见，但它已经失去了焦点，故不可与用户交互。
         */
        Log.e(TAG, "onPause");
        stopBackgroundThread();
        closeCamera();
        super.onPause();
    }

    /**
     * 通过TCP发送中间结果.
     * @param msg
     */
    private void message(float[] msg){
        new SendMessage(msg).execute();
    }

    /**
     * 处理输入image
     * @param bitmap
     * @return the bitmap converted to float
     */
    private float[] getInputData(Bitmap bitmap) {
        final int INPUT_SIDE_LENGTH = 32;

        Mat imageMat = new Mat();

        Utils.bitmapToMat(bitmap, imageMat);

        Imgproc.cvtColor(imageMat, imageMat, Imgproc.COLOR_RGBA2RGB);
        imageMat = scaleAndCenterCrop(imageMat, INPUT_SIDE_LENGTH);
        imageMat.convertTo(imageMat, CvType.CV_32FC3, 1. / 255);
        imageMat = normalize(imageMat,
                new Scalar(0.485, 0.456, 0.406), new Scalar(0.229, 0.224, 0.225));

        float[] inputData = new float[imageMat.width() * imageMat.height() * imageMat.channels()];

        imageMat.get(0, 0, inputData);

        return inputData;
    }

    /**
     * TCP Receiver Helper Function
     */
    class Thread1 implements Runnable{
        public void run(){
            Socket socket = null;
            try{
                // 1. 在端口8887，启动服务，等待其他设备连接
                serverSocket = new ServerSocket(SERVERPORT);
            }catch(IOException e){
                e.printStackTrace();
            }
            // 判断当前线程没有被发送中断请求
            while (!Thread.currentThread().isInterrupted()){
                try{
                    // 2. 有设备连入，获得连入设备的socket
                    socket = serverSocket.accept();
                    // 3. 启动一个新的线程处理远程请求
                    Thread2 commThread = new Thread2(socket);
                    new Thread(commThread).start();
                    // 4. 有设备连入，当前线程退出
                    serverSocket.close();//这里是我自己加的
                    return;
                }catch (IOException e){
                    e.printStackTrace();
                }
            }
        }
    }

    /**
     * TCP Receiver Helper Function
     */
    class Thread2 implements Runnable{
        private Socket clientSocket; // 远程连接的客户端socket
        private BufferedReader input; // clientSocket传输过来的输入数据
        public Thread2(Socket clientSocket){
            // 线程创建时候的初始化
            this.clientSocket = clientSocket;
            try{
                this.input = new BufferedReader(new InputStreamReader(this.clientSocket.getInputStream()));
            }catch (IOException e){
                e.printStackTrace();
            }
        }

        public void run(){
            // 当前线程没有被中断
            while(!Thread.currentThread().isInterrupted()){
                try{
                    // 1. 读取一行数据
                    String read = input.readLine();
                    if(read != null){
                        // 2. 给UI线程发送消息，更新Server time
                        UIHandler.post(new UpdateUIThread(read));
                    }else{
                        // 3. 新开Thread1线程，继续监听
                        Thread1 = new Thread(new Thread1());
                        Thread1.start();
                        return;
                    }
                }catch (IOException e){
                    e.printStackTrace();
                }
            }
        }
    }

    /**
     * TCP Receiver Helper Function
     * 一旦从服务器收到结果后，更新UI并测量时间
     */
    class UpdateUIThread implements Runnable{
        private String msg;
        public UpdateUIThread(String str){
            this.msg = str;
        }

        @Override
        public void run(){
            //  步骤1: 更新lastRecv => 当前时间
            lastRecv = System.currentTimeMillis();
            // 步骤2: 更新界面中Server time 的值 = lastRecv - lastSend
            tvInternet.setText(String.valueOf(lastRecv - lastSend));
            /**
             * “INFO0333”表示从开始发送中间结果到接收结果所花费的时间
             */
            Log.i("INFO0333" , String.valueOf(lastRecv - lastSend));
        }
    }

    /**
     * 网络连接Helper Function
     * 检查网络是否连接
     */
    public boolean isConnected() {
        boolean connected = false;
        try {
            ConnectivityManager cm = (ConnectivityManager) getApplicationContext().getSystemService(Context.CONNECTIVITY_SERVICE);
            NetworkInfo nInfo = cm.getActiveNetworkInfo();
            connected = nInfo != null && nInfo.isAvailable() && nInfo.isConnected();
            return connected;
        } catch (Exception e) {
            Log.e("Connectivity Exception", e.getMessage());
        }
        return connected;
    }

    /**
     * 网络连接 Helper Function
     * @return AP, BSSID 的唯一标志符
     */
    public String getCurrentBssid() {
        Log.i("**** " , "getCurrentBssid is called.");
        String ssid = null;
        // 步骤1: 获取网络连接服务
        ConnectivityManager connManager = (ConnectivityManager) getApplicationContext().getSystemService(Context.CONNECTIVITY_SERVICE);
        // 步骤2: 获取网络连接信息
        NetworkInfo networkInfo = connManager.getNetworkInfo(ConnectivityManager.TYPE_WIFI);
        if (networkInfo.isConnected()) {
            // 步骤3: 获取WIFI服务
            WifiManager wifiManager = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
            // 步骤3: 获取WIFI的BSSID
            ssid = wifiManager.getConnectionInfo().getBSSID();
//            Log.i("****BSSID" , ssid); // ****BSSID: 0c:72:2c:c9:a2:f2
        }
        if(ssid == null){
            // 为空，返回0
            return "0";
        }
        switch(ssid.charAt(15)){
            case 'f':
                return "A";
            case '2':
                return "B";
            case '6':
                return "C";
            default:
                return "X";
        }
    }

    /**
     * Network Connection Helper Function
     * 断开wifi，系统将重试连接.
     */
    public void disconnect() {
        ConnectivityManager connManager = (ConnectivityManager) getApplicationContext().getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo networkInfo = connManager.getNetworkInfo(ConnectivityManager.TYPE_WIFI);
        if (networkInfo.isConnected()) {
            WifiManager wifiManager = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
            wifiManager.disconnect();
        }
    }



    /**
     * OpenCV Helper Function => 将bitmap转换为float数组时使用
     * @param mat
     * @param mean
     * @param std
     * @return Mat
     */
    private Mat normalize(Mat mat, Scalar mean, Scalar std) {
        Mat _mat = mat.clone();
        Core.subtract(_mat, mean, _mat);
        Core.divide(_mat, std, _mat);
        return _mat;
    }

    /**
     * OpenCV Helper Function => 将bitmap转换为float数组时使用
     * @param mat
     * @param sideLength
     * @return Mat
     */
    private Mat scaleAndCenterCrop(Mat mat, int sideLength) {
        Mat _mat = mat.clone();
        double rate;
        if (_mat.height() > _mat.width()) {
            rate = 1. * sideLength / _mat.width();
        } else {
            rate = 1. * sideLength / _mat.height();
        }

        Imgproc.resize(_mat, _mat, new Size(0, 0), rate, rate, Imgproc.INTER_LINEAR);

        if (_mat.height() > _mat.width()) {
            _mat = new Mat(_mat, new Rect(0, (_mat.height() - _mat.width()) / 2, _mat.width(), _mat.width()));
        } else {
            _mat = new Mat(_mat, new Rect((_mat.width() - _mat.height()) / 2, 0, _mat.height(), _mat.height()));
        }
        return _mat;
    }
}
