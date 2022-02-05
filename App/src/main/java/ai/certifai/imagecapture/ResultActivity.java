/*
 * Copyright (c) 2020-2021 CertifAI Sdn. Bhd.
 * This program is part of OSRFramework. You can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version. You should have received a copy of the
 * GNU Affero General Public License along with this program.  If not, see
 * https://www.gnu.org/licenses/agpl-3.0
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 */

package ai.certifai.imagecapture;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.Category;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

import ai.certifai.imagecapture.ml.ModelSkin;
import ai.certifai.imagecapture.utils.ImageReader;

/**
 * Class handling result showing activity
 *
 * @author YCCertifai
 */
public class ResultActivity extends AppCompatActivity
{
    // layouts
    private ImageView image;

    private String fileLoc;

    private EditText label_result;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        // define layouts
        image = findViewById(R.id.resultView);
        label_result = findViewById(R.id.label_result);

        // load image
        File imageFile = getImageFile();
        Bitmap imageBitmap = ImageReader.readFileToImage(imageFile);

        // Tflite Inference
        try {
            ModelSkin model = ModelSkin.newInstance(this);

            // Creates inputs for reference.
            TensorImage image = TensorImage.fromBitmap(imageBitmap);

            // Runs model inference and gets result.
            ModelSkin.Outputs outputs = model.process(image);
            List<Category> probability = outputs.getProbabilityAsCategoryList();

            double maxSoFar = 0;
            String selectedLabel = "";

            // for each loop
            for (Category catDetails : probability)
            {
                if (catDetails.getScore() > maxSoFar)
                {
                    maxSoFar = Math.round(catDetails.getScore() * 100.0) / 100.0;
                    selectedLabel = catDetails.getLabel();
                }
            }

            label_result.setText("This is " + selectedLabel + " (" + maxSoFar * 100 + "%) ");

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

        setImage(imageBitmap);
    }

    private File getImageFile()
    {
        fileLoc = getIntent().getStringExtra("image_file");
        return new File(fileLoc);
    }

    private void setImage(Bitmap imageBitmap)
    {
        image.setImageBitmap(imageBitmap);
        image.setVisibility(View.VISIBLE);
    }
}