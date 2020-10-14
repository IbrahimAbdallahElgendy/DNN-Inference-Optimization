package aflak.me.tensorflowlitexor;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {

    Button buttonToLocalInference;
    Button buttonToCoInference;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        buttonToLocalInference= (Button)findViewById(R.id.btnToLocalInference);
        buttonToCoInference = (Button)findViewById(R.id.btnToCoInference);
    }


    public void doLocalInference(View view){
        Intent intent = new Intent();
        intent.setClass(this, LocalInferenceActivity.class);
        startActivity(intent);
    }

    public void doLayerComputeTime(View view){
        Intent intent = new Intent();
        intent.setClass(this, LayerComputeTimeActivity.class);
        startActivity(intent);
    }

    public void doCoInference(View view){
        Intent intent = new Intent();
        intent.setClass(this, CoInferenceActivity.class);
        startActivity(intent);
    }

    public void doLayerUpTime(View view) {
        Intent intent = new Intent();
        intent.setClass(this, LayerUpTimeActivity.class);
        startActivity(intent);
    }

    public void doLayerDownTime(View view) {
        Intent intent = new Intent();
        intent.setClass(this, LayerDownTimeActivity.class);
        startActivity(intent);
    }
}
