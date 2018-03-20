<?php
$xml = new SimpleXMLElement('<members/>');

//loop through the data, and add each record to the xml object

foreach($jsonFile_decoded AS $members){
	foreach($members AS $memberDetails){
		$member = $xml->addChild('member');
		$member->addChild('lastName', $memberDetails[0]);
		$member->addChild('firstName', $memberDetails[1]);
		$member->addChild('age', $memberDetails[2]);
		$member->addChild('sex', $memberDetails[3]);
		$member->addChild('location', $memberDetails[4]);
	}
}


function be_crm_connector( $fields, $entry, $form_data, $entry_id ) {
        // $api_url = 'http://SalesSimplicity.net/SubmitLead';
        $api_url = 'http://salessimplicity.net/ssnet/svceleads/eleads.asmx?WSDL';


        $client = new SoapClient($api_url)
        $body = array(
                'secret'              => '',
                'name'                => $fields['1']['value'],
                'email'               => $fields['2']['value'],
                'subject'             => $fields['3']['value'],
                'message'             => $fields['4']['value'],
                'date'                => date( 'Y-m-d' ),
        );

        $giud = "DE18C1D5-0707-4E52-B5D4-52EB02836B2B"

        $lead=(object)$body;

        $result = $client->SubmitLead(array('sGUID' => $guid,'Contact' => $lead));

        // Simple error handling
        if ( is_wp_error( $request ) ) {

                $msg  = "There was an error trying to push a lead to the CRM.\n";
                $msg .= 'Error returned: ' . $error = $request->get_error_message() . "\n\n";
                $msg .= "The lead below may need to be added to the CRM manually.\n";
                $msg .= $body['name'] . ' ' . $body['email'];

                wp_mail( get_bloginfo( 'admin_email' ), 'CRM Connector Error', $msg );
        }
         $form_data->skip_mail = true;
}

public function create_new_user_registration($contact_form)
    {
        $wpcf7 = WPCF7_ContactForm::get_current();
        $submission = WPCF7_Submission::get_instance();
        //Below statement will return all data submitted by form.
        $data = $submission->get_posted_data();
        //suppose you have a field which name is 'email' then you can access it by using following statement.
        $user_passed_email =  $data['email'];




    }
?>
