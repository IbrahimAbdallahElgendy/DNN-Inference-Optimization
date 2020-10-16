package aflak.me.tensorflowlitexor;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.support.annotation.RequiresApi;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;

import aflak.me.tensorflowlitexor.util.Constant;

import static aflak.me.tensorflowlitexor.CoInferenceActivity.SERVERPORT;

public class LayerDownTimeActivity extends AppCompatActivity {
    private static final String TAG = "LayerDownTime";

    private TextView tvTime;

    private ServerSocket serverSocket;
    Thread serverThread = null;
    boolean runServerFlag = false;
    String data = "";

    /** 设置端口以接收TCP消息. */
    public static final int SERVERPORT = 8000;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        checkNeedPermissions();
        setContentView(R.layout.activity_layer_down_time);

        tvTime = findViewById(R.id.tv_dt);
    }

    private void checkNeedPermissions(){
        //6.0以上需要动态申请权限
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED
                || ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED
                || ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            //多个权限一起申请
            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE,
                    Manifest.permission.READ_EXTERNAL_STORAGE
            }, 1);
        }
    }

    public void doGoogLeDownTime(View view) {
        /**
         * 启动 TCP Receiver
         */
        this.serverThread  = new ServerThread();
        this.runServerFlag = true;
        this.serverThread.start();
    }

    public void doStopGoogLeDownTime(View view) {
        Log.i(TAG,"doStopGoogLeDownTime called...");
        this.runServerFlag = false;
        saveUtList(LayerDownTimeActivity.this,data);
    }

    /**
     * TCP Receiver Helper Function
     */
    class ServerThread extends Thread{
        public void run(){
            int runNum = 200;
            try{
                // 1. 在端口8888，启动服务，等待其他设备连接
                serverSocket = new ServerSocket(SERVERPORT);
                int recNum = 0;
                double sumTime = 0;
                // 判断当前线程没有被发送中断请求
                while (!Thread.currentThread().isInterrupted()){
                    // 2. 有设备连入，获得连入设备的socket
                    Socket clientSocket = serverSocket.accept();
                    recNum += 1;
                    double t0 = System.nanoTime();//获取纳秒

//                    Log.i(TAG, "connected... " + clientSocket.toString());

                    // 3. 处理远程请求
                    // 装饰流BufferedReader封装输入流（接收客户端的流）
                    BufferedInputStream bis = new BufferedInputStream(
                            clientSocket.getInputStream());

                    DataInputStream dis = new DataInputStream(bis);
                    byte[] bytes = new byte[1024*1024];
                    int dataLen = 0;
                    int len = dis.read(bytes);
                    while (len != -1) {
                        dataLen += len;
                        len = dis.read(bytes);
                    }
                    try {
                        if (clientSocket != null) {
                            clientSocket.close();
                        }
                    } catch (IOException e) {
                        System.out.println(e.getMessage());
                    }

                    // 计时
                    double t1 = System.nanoTime();//获取纳秒
                    double downTime = (t1 - t0)/(1000000.0);
                    sumTime += downTime;
                    if(recNum == runNum){
                        double meanTime = sumTime / runNum;
                        meanTime = new BigDecimal(meanTime).setScale(4, BigDecimal.ROUND_HALF_UP).doubleValue();
                        data = data + (dataLen/4) + "\t" + meanTime + "\n";
                        recNum = 0;
                        sumTime = 0;
                    }
                    Log.i(TAG, "recv len " + dataLen + " time " + downTime);

                    if(! runServerFlag){
                        saveUtList(LayerDownTimeActivity.this,data);
                        return;
                    }
                }

            }catch(IOException e){
                e.printStackTrace();
            }finally {
                if(serverSocket != null){
                    try{
                        serverSocket.close();
                    }catch(IOException e){
                        e.printStackTrace();
                    }
                }
            }

        }
    }

    /**
     * 保存文件
     * @param context
     * @param data
     * @return
     */
    public static boolean saveUtList(Context context, String data) {
        Log.i(TAG,"saveUtList called ....");

        try {
            // 判断当前的手机是否有sd卡
            String state = Environment.getExternalStorageState();

            if(!Environment.MEDIA_MOUNTED.equals(state)) {
                // 已经挂载了sd卡
                return false;
            }

            File sdCardFile = Environment.getExternalStorageDirectory();
            File file = new File(sdCardFile, "MobileNodeDownloadTime.txt");

            FileOutputStream fos = new FileOutputStream(file);

            fos.write(data.getBytes());

            fos.flush();
            fos.close();
            return true;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

}
